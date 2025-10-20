import json
import time
import paho.mqtt.client as mqtt
from orchestrator.config import DEPLOYMENT_PLAN_PATH
import threading

# Public EMQX broker that works via HTTPS/WebSockets
BROKER = "broker.emqx.io"
PORT = 8084  # WebSocket + TLS (firewall-friendly)
acks = {}
acks_lock = threading.Lock()
TIMEOUT = 30 
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(" Orchestrator connected to MQTT broker")
        # Subscribe to ACKs from any site
        client.subscribe("fmaas/ack/#")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    print(f"ACK received on {msg.topic}: {msg.payload.decode()}")

def deploy_to_sites(plan_path=DEPLOYMENT_PLAN_PATH):
    with open(plan_path, "r") as f:
        plan = json.load(f)

    client = mqtt.Client(client_id="orchestrator", transport="websockets")
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    site_ids = [site["id"] for site in plan["sites"]]
    print(f"Deploying to sites: {site_ids}")

    # Publish deployment messages to each site
    for site in plan["sites"]:
        topic = f"fmaas/deploy/site/{site['id']}"
        msg = json.dumps(site["deployments"])
        print(f"Deploying to {topic}: {msg}")
        client.publish(topic, msg)
        time.sleep(0.5)

    # Wait for all ACKs
    print(f"Waiting up to {TIMEOUT}s for acknowledgements...")
    print(site_ids)
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        with acks_lock:
            if all(site_id in acks for site_id in site_ids):
                break
        time.sleep(1)

    # Stop MQTT loop
    client.loop_stop()
    client.disconnect()

    # Print ACK summary
    print("\n Deployment summary:")
    print(site_ids)
    for site_id in site_ids:
        if site_id in acks:
            print(f" {site_id} ACK: {acks[site_id]}")
        else:
            print(f" No ACK from {site_id} within {TIMEOUT}s")

    print("All deployments published and ACKs received (if any).")

if __name__ == "__main__":
    deploy_to_sites()
