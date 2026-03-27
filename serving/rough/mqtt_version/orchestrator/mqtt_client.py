"""MQTT connection management and ACK tracking for the orchestrator."""

import json
import ssl
import threading
import time
import paho.mqtt.client as mqtt
from orchestrator.config import BROKER, PORT


class MQTTManager:
    """Manages a single MQTT connection with ACK tracking.

    Handles TLS setup, subscriptions, and waiting for ACKs from site managers.
    Each method that needs MQTT creates/reuses a client via _make_client().
    """

    def __init__(self):
        self._acks = {}
        self._acks_lock = threading.Lock()
        self._all_acks_event = threading.Event()
        self._connected_event = threading.Event()
        self._expected_sites = set()
        self._waiting_for_ack_type = None
        self._active_client = None  # kept so reset_acks can flush stale messages

    def _make_client(self, client_id: str) -> mqtt.Client:
        """Create and configure a new MQTT client with TLS."""
        # clean_session=True: broker discards any queued messages from previous
        # sessions for this client ID, preventing stale ACK replays on reconnect.
        client = mqtt.Client(client_id=client_id, transport="websockets", clean_session=True)
        client.enable_logger()
        try:
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
        except Exception as e:
            print(f"Warning: TLS setup failed: {e}")
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        return client

    def connect(self, client_id: str) -> mqtt.Client:
        """Create, connect, and return a ready MQTT client."""
        client = self._make_client(client_id)
        print(f"Connecting to {BROKER}:{PORT} ...")
        try:
            client.connect(BROKER, PORT, 60)
        except Exception as e:
            print(f"ERROR: Failed to connect: {e}")
            raise
        client.loop_start()
        if not self._connected_event.wait(timeout=10):
            raise RuntimeError("MQTT not connected/subscribed in time.")
        self._connected_event.clear()  # Reset for next connection
        self._active_client = client
        return client

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Orchestrator connected to MQTT broker")
            client.subscribe("fmaas/deploytime/ack/#", qos=1)
            client.subscribe("fmaas/runtime/ack/#", qos=1)
            client.subscribe("fmaas/cleanup/ack/#", qos=1)
            self._connected_event.set()
        else:
            error_map = {
                1: "MQTT_ERR_PROTOCOL_VERSION",
                2: "MQTT_ERR_INVALID_CLIENT_ID",
                3: "MQTT_ERR_SERVER_UNAVAILABLE",
                4: "MQTT_ERR_BAD_USERNAME_PASSWORD",
                5: "MQTT_ERR_NOT_AUTHORIZED",
            }
            print(f"MQTT connection failed (code {rc}: {error_map.get(rc, f'Unknown {rc}')})")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return

        waiting_for = self._waiting_for_ack_type

        if topic.startswith("fmaas/deploytime/ack"):
            site = payload.get("site", "unknown")
            print(f"Deploytime ACK from {site}: {payload}")
            # Runtime deployment ACKs use ack_id and must still be recorded even
            # while the orchestrator is in the long-lived runtime phase.
            if payload.get("ack_id") or waiting_for == 'deploytime' or waiting_for is None:
                self._record_ack(site, payload)
        elif topic.startswith("fmaas/runtime/ack"):
            site = payload.get("site", "unknown")
            if waiting_for == 'runtime' or waiting_for is None:
                self._record_ack(site, payload)
            print(f"Runtime ACK from {site}")
        elif topic.startswith("fmaas/cleanup/ack"):
            site = payload.get("site", "unknown")
            if waiting_for == 'cleanup' or waiting_for is None:
                self._record_ack(site, payload)
            print(f"Cleanup ACK from {site}")

    def _record_ack(self, site: str, payload: dict):
        ack_key = payload.get("ack_id") or site
        with self._acks_lock:
            self._acks[ack_key] = payload
            if self._expected_sites and all(s in self._acks for s in self._expected_sites):
                self._all_acks_event.set()

    def reset_acks(self, expected_sites: set, ack_type: str = None):
        """Prepare to wait for a new round of ACKs."""
        self._acks.clear()
        self._all_acks_event.clear()
        self._expected_sites = expected_sites
        self._waiting_for_ack_type = ack_type

    def wait_for_acks(self, timeout: float) -> bool:
        """Block until all expected ACKs arrive or timeout. Returns True if all received."""
        received = self._all_acks_event.wait(timeout=timeout)
        self._waiting_for_ack_type = None
        return received

    def wait_for_any_ack(self, expected_keys: set, timeout: float):
        """Return the first matching ACK key/payload that arrives within timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._acks_lock:
                for key in expected_keys:
                    if key in self._acks:
                        return key, self._acks[key]
            time.sleep(0.05)
        return None, None
