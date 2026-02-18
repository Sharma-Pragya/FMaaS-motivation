import json
import asyncio
import os
import ssl
import threading
import traceback

import paho.mqtt.client as mqtt

from site_manager.config import BROKER, PORT, SITE_ID
from site_manager.storage import (
    store_plan, store_requests, get_requests, get_deployments,
    get_output_dir, clear_state, append_deployments,
    mark_task_deploying, mark_task_deployed
)
from site_manager.runtime_executor import handle_runtime_request_continuous, initialize_dataloaders
from site_manager.deployment_handler import deploy_models, shutdown_devices, _add_decoder_to_device


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Site {SITE_ID} connected to broker")
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/req", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/add", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/update", qos=1)
        client.subscribe(f"fmaas/runtime/start/site/{SITE_ID}", qos=1)
        client.subscribe(f"fmaas/cleanup/site/{SITE_ID}", qos=1)
    else:
        print(f"[MQTT] Connection failed with code {rc}")


def _save_results(reqs_latency):
    """Save request_latency_results.csv and model_deployment_results.json.
    
    Saves to the output_dir specified by the orchestrator (if provided),
    otherwise saves to the current working directory.
    """
    output_dir = get_output_dir()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "request_latency_results.csv")
    else:
        csv_path = "request_latency_results.csv"

    # Save latency CSV
    reqs = get_requests()
    reqs_dict = {req['req_id']: req for req in reqs}
    
    try:
        with open(csv_path, "w") as f:
            f.write("req_id,req_time,site_manager,device,backbone,task,"
                    "site_manager_send_time,device_start_time,device_end_time,"
                    "end_to_end_latency(ms),proc_time(ms),swap_time(ms),decoder_time(ms),"
                    "pred,true\n")
            for entry in reqs_latency:
                (req_id, device_url, site_manager_send_time, device_start_time,
                device_end_time, e2e_latency, proc_time, swap_time, decoder_time, pred, true_val) = entry
                req = reqs_dict.get(req_id, {})
                req_time = req.get('req_time', -1)
                backbone = req.get('backbone', 'unknown')
                task = req.get('task', 'unknown')
                f.write(f"{req_id},{req_time},{SITE_ID},{device_url},{backbone},"
                        f"{task},{site_manager_send_time},{device_start_time},"
                        f"{device_end_time},{e2e_latency*1000},{proc_time*1000},{swap_time*1000},{decoder_time*1000},"
                        f"{pred},{true_val}\n")

        print(f"[SiteManager] Saved latency results to {csv_path} "
            f"({len(reqs_latency)} entries)")
        return csv_path
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")
        traceback.print_exc()
        raise


def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())
    except Exception as e:
        print(f"[MQTT] Failed to parse message on {topic}: {e}")
        return

    # ── New deployment: clear old state first ────────────────────────────
    if topic.endswith(f"deploy/site/{SITE_ID}"):
        print(f"[MQTT] Received deployment plan for {SITE_ID}")
        try:
            # Clear state from any previous experiment
            clear_state()
            store_plan(payload)

            output_dir = get_output_dir()
            print(f"[SiteManager] output_dir = {output_dir}")

            deployment_status = asyncio.run(deploy_models(payload["deployments"]))

            # Save deployment status
            json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(deployment_status, f, indent=4)
            print(f"[SiteManager] Saved deployment results to {json_path}")

            client.publish(
                f"fmaas/deploytime/ack/site/{SITE_ID}",
                json.dumps({'site': SITE_ID, 'deploymentstatus': deployment_status}),
                qos=1
            )
            print(f"[MQTT] Sent deploytime ACK for {SITE_ID}")

        except Exception as e:
            print(f"[SiteManager] ERROR during deployment: {e}")
            traceback.print_exc()
            # Still send ACK so orchestrator doesn't hang forever
            client.publish(
                f"fmaas/deploytime/ack/site/{SITE_ID}",
                json.dumps({'site': SITE_ID, 'error': str(e)}),
                qos=1
            )

    # ── Runtime add: full new deployment (new server or new backbone) ──────
    elif topic.endswith(f"deploy/site/{SITE_ID}/add"):
        print(f"[MQTT] Received /add for {SITE_ID}")

        # Run deployment in background thread to avoid blocking MQTT message reception
        def deploy_in_background():
            try:
                new_specs = payload.get("deployments", [payload] if isinstance(payload, dict) and "device" in payload else [])
                if not new_specs:
                    raise ValueError("No deployments in /add payload")

                # Extract task names and mark them as deploying BEFORE starting
                task_names = []
                for spec in new_specs:
                    for dec in spec.get("decoders", []):
                        tname = dec.get("task")
                        if tname:
                            task_names.append(tname)
                            mark_task_deploying(tname)

                print(f"[SiteManager] Starting background deployment for /add (deploying tasks: {task_names})...")
                deployment_status = asyncio.run(deploy_models(new_specs))
                append_deployments(new_specs)

                # Mark all tasks as deployed now that deployment is complete
                for tname in task_names:
                    mark_task_deployed(tname)

                # Append to model_deployment_results.json
                output_dir = get_output_dir()
                json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
                try:
                    # Read existing deployment results
                    if os.path.exists(json_path):
                        with open(json_path, "r") as f:
                            deployment_results = json.load(f)
                    else:
                        deployment_results = []

                    # Append runtime add event
                    for status in deployment_status:
                        deployment_results.append({
                            "event": "runtime_add",
                            "deployments": new_specs,
                            "result": status
                        })

                    # Write back
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    with open(json_path, "w") as f:
                        json.dump(deployment_results, f, indent=4)
                    print(f"[SiteManager] Updated deployment results: {json_path}")
                except Exception as e:
                    print(f"[SiteManager] Warning: Failed to update deployment JSON: {e}")

                print(f"[SiteManager] Background deployment complete for /add")
            except Exception as e:
                print(f"[SiteManager] ERROR during background /add deployment: {e}")
                traceback.print_exc()
                # Unblock requests even on failure so they aren't deferred forever
                for tname in task_names:
                    mark_task_deployed(tname)

        threading.Thread(target=deploy_in_background, daemon=True).start()

        # Send ACK immediately (don't wait for deployment to complete)
        client.publish(
            f"fmaas/deploytime/ack/site/{SITE_ID}",
            json.dumps({"site": SITE_ID, "add": True, "status": "started"}),
            qos=1,
        )
        print(f"[MQTT] Sent /add ACK for {SITE_ID} (deployment running in background)")

    # ── Runtime update: hot-add decoder to existing device ─────────────────
    elif topic.endswith(f"deploy/site/{SITE_ID}/update"):
        print(f"[MQTT] Received /update for {SITE_ID}")

        # Run deployment in background thread to avoid blocking MQTT message reception
        def deploy_in_background():
            try:
                device_url = payload.get("device")
                decoders = payload.get("decoders", [])
                if not device_url or not decoders:
                    raise ValueError("device and decoders required in /update payload")

                # Extract task names and mark them as deploying BEFORE starting
                task_names = [dec.get("task") for dec in decoders if dec.get("task")]
                for tname in task_names:
                    mark_task_deploying(tname)

                print(f"[SiteManager] Starting background deployment for /update (deploying tasks: {task_names})...")
                result = asyncio.run(_add_decoder_to_device(device_url, decoders))

                # Mark all tasks as deployed now that deployment is complete
                for tname in task_names:
                    mark_task_deployed(tname)

                # Append to model_deployment_results.json
                output_dir = get_output_dir()
                json_path = os.path.join(output_dir, "model_deployment_results.json") if output_dir else "model_deployment_results.json"
                try:
                    # Read existing deployment results
                    if os.path.exists(json_path):
                        with open(json_path, "r") as f:
                            deployment_results = json.load(f)
                    else:
                        deployment_results = []

                    # Append runtime update event
                    deployment_results.append({
                        "event": "runtime_update",
                        "device": device_url,
                        "result": result
                    })

                    # Write back
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    with open(json_path, "w") as f:
                        json.dump(deployment_results, f, indent=4)
                    print(f"[SiteManager] Updated deployment results: {json_path}")
                except Exception as e:
                    print(f"[SiteManager] Warning: Failed to update deployment JSON: {e}")

                print(f"[SiteManager] Background deployment complete for /update")
            except Exception as e:
                print(f"[SiteManager] ERROR during background /update deployment: {e}")
                traceback.print_exc()
                # Unblock requests even on failure so they aren't deferred forever
                for tname in task_names:
                    mark_task_deployed(tname)

        threading.Thread(target=deploy_in_background, daemon=True).start()

        # Send ACK immediately (don't wait for deployment to complete)
        client.publish(
            f"fmaas/deploytime/ack/site/{SITE_ID}",
            json.dumps({"site": SITE_ID, "update": True, "status": "started"}),
            qos=1,
        )
        print(f"[MQTT] Sent /update ACK for {SITE_ID} (deployment running in background)")

    # ── Receive request chunks ───────────────────────────────────────────
    elif topic.endswith(f"deploy/site/{SITE_ID}/req"):
        print(f"[MQTT] Received request chunk for {SITE_ID}")
        store_requests(payload)

    # ── Runtime start: execute trace and save results ────────────────────
    elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
        print(f"[MQTT] Start signal received for {SITE_ID}")

        # Run inference in a separate thread so MQTT callbacks aren't blocked
        def run_inference():
            try:
                output_dir = get_output_dir()
                print(f"[SiteManager] Starting continuous inference mode, output_dir={output_dir}")

                # Use continuous mode that picks up new requests dynamically
                reqs_latency = asyncio.run(handle_runtime_request_continuous())

                # Save results to the orchestrator-specified output directory
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
                    qos=1
                )
                print(f"[MQTT] Sent runtime ACK for {SITE_ID}")

            except Exception as e:
                print(f"[SiteManager] ERROR during runtime: {e}")
                traceback.print_exc()
                # Still send ACK so orchestrator doesn't hang forever
                client.publish(
                    f"fmaas/runtime/ack/site/{SITE_ID}",
                    json.dumps({"site": SITE_ID, "status": "error", "error": str(e)}),
                    qos=1
                )

        threading.Thread(target=run_inference, daemon=True).start()
        print(f"[MQTT] Started continuous inference in background thread (non-blocking)")

    # ── Cleanup: kill Triton servers on devices ──────────────────────────
    elif topic.endswith(f"cleanup/site/{SITE_ID}"):
        print(f"[MQTT] Cleanup signal received for {SITE_ID}")
        try:
            deployments = get_deployments()
            if deployments:
                asyncio.run(shutdown_devices(deployments))
        except Exception as e:
            print(f"[SiteManager] Cleanup error: {e}")
            traceback.print_exc()

        client.publish(
            f"fmaas/cleanup/ack/site/{SITE_ID}",
            json.dumps({"site": SITE_ID, "status": "cleaned"}),
            qos=1
        )
        print(f"[MQTT] Sent cleanup ACK for {SITE_ID}")


def start_site_mqtt_agent():
    import socket
    client_id = f"{SITE_ID}-{socket.gethostname()}-{os.getpid()}"
    client = mqtt.Client(client_id=client_id, transport="websockets",clean_session=True)
    client.enable_logger()
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 300)

    # Start MQTT network loop in background thread so callbacks don't block message reception
    client.loop_start()

    initialize_dataloaders()
    print(f"[SiteManager] {SITE_ID} ready. Entering MQTT loop")

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
