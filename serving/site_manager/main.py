import asyncio
import os
import json
import time
import ssl
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager

# File lock for safe concurrent reads/writes to model_deployment_results.json
_deployment_json_lock = threading.Lock()

@contextmanager
def _deployment_json(json_path, output_dir):
    """Read, yield, write model_deployment_results.json under a lock."""
    with _deployment_json_lock:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        yield data
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

import paho.mqtt.client as mqtt

from site_manager.config import BROKER, PORT, SITE_ID
from site_manager.deployment_handler import (
    _add_decoder_to_device,
    _swap_backbone_on_device,
    deploy_models,
    shutdown_devices,
)
from site_manager.runtime_executor import handle_runtime_request_continuous, initialize_dataloaders
from site_manager.storage import (
    append_deployments,
    clear_state,
    get_deployments,
    get_output_dir,
    get_requests,
    replace_deployment,
    store_plan,
    store_requests,
)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Site {SITE_ID} connected to broker")
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/req", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/add", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/update", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/migrate", qos=1)
        client.subscribe(f"fmaas/runtime/start/site/{SITE_ID}", qos=1)
        client.subscribe(f"fmaas/cleanup/site/{SITE_ID}", qos=1)
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def _publish_runtime_deploy_ack(client, ack_id, status, action, extra=None):
    payload = {
        "site": SITE_ID,
        "ack_id": ack_id,
        "status": status,
        "action": action,
    }
    if extra:
        payload.update(extra)
    client.publish(
        f"fmaas/deploytime/ack/site/{SITE_ID}",
        json.dumps(payload),
        qos=1,
    )
    print(f"[MQTT] Sent runtime deploy ACK for {SITE_ID}: {payload}")


def _save_results(reqs_latency):
    """Preserve the old site_manager result outputs and timing summary."""
    output_dir = get_output_dir()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "request_latency_results.csv")
        summary_path = os.path.join(output_dir, "serving_timing_summary.json")
    else:
        csv_path = "request_latency_results.csv"
        summary_path = "serving_timing_summary.json"

    reqs = get_requests()
    reqs_dict = {req["req_id"]: req for req in reqs}
    valid_entries = [entry for entry in reqs_latency if entry is not None]

    per_device = defaultdict(list)
    for entry in valid_entries:
        per_device[entry[1]].append(entry)

    timing_summary = {"devices": {}}
    for device_url, entries in per_device.items():
        entries.sort(key=lambda item: item[4])
        count = len(entries)
        prep_ms = sum((item[3] - item[2]) * 1000.0 for item in entries) / count
        submit_to_backend_ms = sum((item[4] - item[3]) * 1000.0 for item in entries) / count
        backend_exec_ms = sum((item[5] - item[4]) * 1000.0 for item in entries) / count
        backend_to_client_ms = sum((item[6] - item[5]) * 1000.0 for item in entries) / count
        e2e_ms = sum(item[7] * 1000.0 for item in entries) / count

        start_to_start_ms = None
        idle_gap_ms = None
        overlap_count = 0
        if count > 1:
            start_to_start_samples = []
            idle_gap_samples = []
            prev_start = entries[0][4]
            prev_end = entries[0][5]
            for item in entries[1:]:
                start_to_start_samples.append((item[4] - prev_start) * 1000.0)
                gap_ms = (item[4] - prev_end) * 1000.0
                if gap_ms < 0:
                    overlap_count += 1
                    gap_ms = 0.0
                idle_gap_samples.append(gap_ms)
                prev_start = item[4]
                prev_end = item[5]
            start_to_start_ms = sum(start_to_start_samples) / len(start_to_start_samples)
            idle_gap_ms = sum(idle_gap_samples) / len(idle_gap_samples)

        timing_summary["devices"][device_url] = {
            "request_count": count,
            "avg_client_prep_ms": prep_ms,
            "avg_client_submit_to_backend_start_ms": submit_to_backend_ms,
            "avg_backend_exec_ms": backend_exec_ms,
            "avg_backend_to_client_return_ms": backend_to_client_ms,
            "avg_end_to_end_ms": e2e_ms,
            "avg_backend_start_to_start_ms": start_to_start_ms,
            "avg_backend_idle_gap_ms": idle_gap_ms,
            "backend_overlap_pairs": overlap_count,
        }

    with open(csv_path, "w") as f:
        f.write(
            "req_id,req_time,site_manager,device,backbone,task,"
            "site_manager_send_time,client_infer_submit_time,device_start_time,device_end_time,client_receive_time,"
            "client_prep_time(ms),client_submit_to_backend_start(ms),backend_exec_time(ms),backend_to_client_return(ms),"
            "end_to_end_latency(ms),proc_time(ms),swap_time(ms),decoder_time(ms),"
            "pred,true\n"
        )
        for entry in valid_entries:
            (
                req_id,
                device_url,
                site_manager_send_time,
                client_infer_submit_time,
                device_start_time,
                device_end_time,
                client_receive_time,
                e2e_latency,
                proc_time,
                swap_time,
                decoder_time,
                pred,
                true_val,
            ) = entry
            req = reqs_dict.get(req_id, {})
            req_time = req.get("req_time", -1)
            backbone = req.get("backbone", "unknown")
            task = req.get("task", "unknown")
            client_prep_ms = (client_infer_submit_time - site_manager_send_time) * 1000.0
            submit_to_backend_ms = (device_start_time - client_infer_submit_time) * 1000.0
            backend_exec_ms = (device_end_time - device_start_time) * 1000.0
            backend_to_client_ms = (client_receive_time - device_end_time) * 1000.0
            f.write(
                f"{req_id},{req_time},{SITE_ID},{device_url},{backbone},{task},"
                f"{site_manager_send_time},{client_infer_submit_time},{device_start_time},{device_end_time},{client_receive_time},"
                f"{client_prep_ms},{submit_to_backend_ms},{backend_exec_ms},{backend_to_client_ms},{e2e_latency*1000},"
                f"{proc_time*1000},{swap_time*1000},{decoder_time*1000},{pred},{true_val}\n"
            )

    with open(summary_path, "w") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"[SiteManager] Saved latency results to {csv_path} ({len(valid_entries)} entries)")
    print(f"[SiteManager] Saved serving timing summary to {summary_path}")
    return csv_path


def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())

        if topic.endswith(f"deploy/site/{SITE_ID}"):
            print(f"[MQTT] Received deployment plan for {SITE_ID}")
            clear_state()
            store_plan(payload)
            deployment_status = asyncio.run(deploy_models(payload.get("deployments", [])))
            output_dir = get_output_dir()
            print(f"[SiteManager] output_dir = {output_dir}")
            json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(deployment_status, f, indent=4)
            print(f"[SiteManager] Saved deployment results to {json_path}")
            client.publish(
                f"fmaas/deploytime/ack/site/{SITE_ID}",
                json.dumps({"site": SITE_ID, "deploymentstatus": deployment_status}),
                qos=1,
            )
            print(f"[MQTT] Sent deploytime ACK for {SITE_ID}")
        elif topic.endswith(f"deploy/site/{SITE_ID}/req"):
            store_requests(payload)
        elif topic.endswith(f"deploy/site/{SITE_ID}/add"):
            print(f"[MQTT] Received /add for {SITE_ID}")

            def deploy_in_background():
                ack_id = payload.get("ack_id")
                try:
                    new_specs = payload.get(
                        "deployments",
                        [payload] if isinstance(payload, dict) and "device" in payload else [],
                    )
                    if not new_specs:
                        raise ValueError("No deployments in /add payload")

                    print(f"[SiteManager] Starting background deployment for /add...")
                    deployment_status = asyncio.run(deploy_models(new_specs))
                    append_deployments(new_specs)
                    output_dir = get_output_dir()
                    json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
                    with _deployment_json(json_path, output_dir) as data:
                        for entry in deployment_status:
                            data.append(entry)
                    _publish_runtime_deploy_ack(
                        client,
                        ack_id,
                        "completed",
                        "add",
                        {"deployments": len(new_specs), "deploymentstatus": deployment_status},
                    )
                except Exception as exc:
                    print(f"[SiteManager] ERROR during background /add deployment: {exc}")
                    traceback.print_exc()
                    _publish_runtime_deploy_ack(client, ack_id, "error", "add", {"error": str(exc)})

            threading.Thread(target=deploy_in_background, daemon=True).start()
        elif topic.endswith(f"deploy/site/{SITE_ID}/update"):
            print(f"[MQTT] Received /update for {SITE_ID}")

            def update_in_background():
                ack_id = payload.get("ack_id")
                try:
                    device_url = payload.get("device")
                    decoders = payload.get("decoders", [])
                    if not device_url or not decoders:
                        raise ValueError("device and decoders required in /update payload")
                    result = asyncio.run(_add_decoder_to_device(device_url, decoders))
                    output_dir = get_output_dir()
                    json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
                    with _deployment_json(json_path, output_dir) as data:
                        data.append(result)
                    _publish_runtime_deploy_ack(
                        client,
                        ack_id,
                        "completed",
                        "update",
                        {"device": device_url, "decoders": len(decoders), "result": result},
                    )
                except Exception as exc:
                    print(f"[SiteManager] ERROR during /update: {exc}")
                    traceback.print_exc()
                    _publish_runtime_deploy_ack(client, ack_id, "error", "update", {"error": str(exc)})

            threading.Thread(target=update_in_background, daemon=True).start()
        elif topic.endswith(f"deploy/site/{SITE_ID}/migrate"):
            print(f"[MQTT] Received /migrate for {SITE_ID}")

            def migrate_in_background():
                ack_id = payload.get("ack_id")
                try:
                    new_specs = payload.get("deployments", [])
                    old_backbone = payload.get("old_backbone")
                    if not new_specs or not old_backbone:
                        raise ValueError("deployments and old_backbone required in /migrate payload")
                    new_spec = new_specs[0]
                    result = asyncio.run(
                        _swap_backbone_on_device(
                            new_spec["device"],
                            new_spec["backbone"],
                            new_spec.get("decoders", []),
                        )
                    )
                    replace_deployment(old_backbone, new_spec)
                    output_dir = get_output_dir()
                    json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
                    with _deployment_json(json_path, output_dir) as data:
                        data.append(result)
                    _publish_runtime_deploy_ack(
                        client,
                        ack_id,
                        "completed",
                        "migrate",
                        {
                            "device": new_spec["device"],
                            "old_backbone": old_backbone,
                            "new_backbone": new_spec["backbone"],
                            "result": result,
                        },
                    )
                except Exception as exc:
                    print(f"[SiteManager] ERROR during /migrate: {exc}")
                    traceback.print_exc()
                    _publish_runtime_deploy_ack(client, ack_id, "error", "migrate", {"error": str(exc)})

            threading.Thread(target=migrate_in_background, daemon=True).start()
        elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
            print(f"[MQTT] Start signal received for {SITE_ID}")

            def run_inference():
                try:
                    output_dir = get_output_dir()
                    print(f"[SiteManager] Starting continuous inference mode, output_dir={output_dir}")
                    reqs_latency, _ = asyncio.run(handle_runtime_request_continuous())
                    csv_path = _save_results(reqs_latency)

                    ack_payload = {
                        "site": SITE_ID,
                        "status": "completed",
                        "total_requests": len(reqs_latency),
                        "results_path": csv_path,
                    }
                    client.publish(
                        f"fmaas/runtime/ack/site/{SITE_ID}",
                        json.dumps(ack_payload),
                        qos=1,
                    )
                    print(f"[MQTT] Sent runtime ACK for {SITE_ID}")
                except Exception as exc:
                    print(f"[SiteManager] ERROR during runtime: {exc}")
                    traceback.print_exc()
                    client.publish(
                        f"fmaas/runtime/ack/site/{SITE_ID}",
                        json.dumps({"site": SITE_ID, "status": "error", "error": str(exc)}),
                        qos=1,
                    )

            threading.Thread(target=run_inference, daemon=True).start()
            print(f"[MQTT] Started continuous inference in background thread (non-blocking)")
        elif topic.endswith(f"cleanup/site/{SITE_ID}"):
            print(f"[MQTT] Received cleanup for {SITE_ID}")

            def cleanup_in_background():
                try:
                    asyncio.run(shutdown_devices(get_deployments()))
                    clear_state()
                    print(f"[SiteManager] Cleanup complete for {SITE_ID}")
                except Exception as exc:
                    print(f"[SiteManager] ERROR during cleanup: {exc}")
                    traceback.print_exc()

            threading.Thread(target=cleanup_in_background, daemon=True).start()
    except Exception as exc:
        print(f"[MQTT] Error while handling topic {topic}: {exc}")
        traceback.print_exc()
        if topic.endswith(f"deploy/site/{SITE_ID}"):
            client.publish(
                f"fmaas/deploytime/ack/site/{SITE_ID}",
                json.dumps({"site": SITE_ID, "error": str(exc)}),
                qos=1,
            )
        elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
            client.publish(
                f"fmaas/runtime/ack/site/{SITE_ID}",
                json.dumps({"site": SITE_ID, "status": "error", "error": str(exc)}),
                qos=1,
            )


def start_site_mqtt_agent():
    import socket

    client_id = f"{SITE_ID}-{socket.gethostname()}-{os.getpid()}"
    client = mqtt.Client(client_id=client_id, transport="websockets", clean_session=True)
    client.enable_logger()
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 300)

    # Start MQTT network loop in background thread so callbacks don't block message reception
    client.loop_start()

    initialize_dataloaders()
    print(f"[SiteManager] {SITE_ID} ready. Entering MQTT loop (custom gRPC transport, legacy control flow)")

    # Keep main thread alive
    try:
        # Wait forever (or until interrupted)
        threading.Event().wait()
    except KeyboardInterrupt:
        print(f"[SiteManager] Shutting down...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    start_site_mqtt_agent()
