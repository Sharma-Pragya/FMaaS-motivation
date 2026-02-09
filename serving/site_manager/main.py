import json, asyncio, time, os
import paho.mqtt.client as mqtt
from site_manager.config import BROKER, PORT, SITE_ID
from site_manager.storage import (
    store_plan, store_requests, get_requests, get_deployments,
    get_output_dir, clear_state
)
from site_manager.runtime_executor import handle_runtime_request, initialize_dataloaders
from site_manager.deployment_handler import deploy_models, shutdown_devices
import ssl


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Site {SITE_ID} connected to broker")
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}", qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/req", qos=1)
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
        json_path = os.path.join(output_dir, "model_deployment_results.json")
    else:
        csv_path = "request_latency_results.csv"
        json_path = "model_deployment_results.json"

    # Save latency CSV
    reqs = get_requests()
    reqs_dict = {req['req_id']: req for req in reqs}
    print(f"[DEBUG] reqs_dict keys: {list(reqs_dict.keys())[:5]}")
    print(f"[DEBUG] reqs_latency first entry req_id: {reqs_latency[0][0] if reqs_latency else 'EMPTY'}")
    print(f"[DEBUG] Total requests in dict: {len(reqs_dict)}, latency entries: {len(reqs_latency)}")
    
    try:
        with open(csv_path, "w") as f:
            f.write("req_id,req_time,site_manager,device,backbone,task,"
                    "site_manager_send_time,device_start_time,"
                    "end_to_end_latency(ms),proc_time(ms),swap_time(ms),"
                    "pred,true\n")
            print(reqs_latency)
            for entry in reqs_latency:
                (req_id, device_url, site_manager_send_time, device_start_time,
                e2e_latency, proc_time, swap_time, pred, true_val) = entry
                req = reqs_dict.get(req_id, {})
                req_time = req.get('req_time', -1)
                backbone = req.get('backbone', 'unknown')
                task = req.get('task', 'unknown')
                f.write(f"{req_id},{req_time},{SITE_ID},{device_url},{backbone},"
                        f"{task},{site_manager_send_time},{device_start_time},"
                        f"{e2e_latency*1000},{proc_time*1000},{swap_time*1000},"
                        f"{pred},{true_val}\n")

        print(f"[SiteManager] Saved latency results to {csv_path} "
            f"({len(reqs_latency)} entries)")
        return csv_path
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")
        import traceback; traceback.print_exc()
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

            try:
                loop = asyncio.get_running_loop()
                deployment_status = loop.create_task(deploy_models(payload["deployments"]))
            except RuntimeError:
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
            import traceback; traceback.print_exc()
            # Still send ACK so orchestrator doesn't hang forever
            client.publish(
                f"fmaas/deploytime/ack/site/{SITE_ID}",
                json.dumps({'site': SITE_ID, 'error': str(e)}),
                qos=1
            )

    # ── Receive request chunks ───────────────────────────────────────────
    elif topic.endswith(f"deploy/site/{SITE_ID}/req"):
        print(f"[MQTT] Received request chunk for {SITE_ID}")
        store_requests(payload)

    # ── Runtime start: execute trace and save results ────────────────────
    elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
        print(f"[MQTT] Start signal received for {SITE_ID}")
        try:
            reqs = get_requests()
            output_dir = get_output_dir()
            print(f"[SiteManager] Runtime: {len(reqs)} requests, output_dir={output_dir}")

            reqs_latency = asyncio.run(execute_cached_requests(client))

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
            import traceback; traceback.print_exc()
            # Still send ACK so orchestrator doesn't hang forever
            client.publish(
                f"fmaas/runtime/ack/site/{SITE_ID}",
                json.dumps({"site": SITE_ID, "status": "error", "error": str(e)}),
                qos=1
            )

    # ── Cleanup: kill Triton servers on devices ──────────────────────────
    elif topic.endswith(f"cleanup/site/{SITE_ID}"):
        print(f"[MQTT] Cleanup signal received for {SITE_ID}")
        try:
            deployments = get_deployments()
            if deployments:
                asyncio.run(shutdown_devices(deployments))
        except Exception as e:
            print(f"[SiteManager] Cleanup error: {e}")
            import traceback; traceback.print_exc()

        client.publish(
            f"fmaas/cleanup/ack/site/{SITE_ID}",
            json.dumps({"site": SITE_ID, "status": "cleaned"}),
            qos=1
        )
        print(f"[MQTT] Sent cleanup ACK for {SITE_ID}")

async def execute_cached_requests(client):
    start = time.time()
    reqs = get_requests()
    reqs_latency =await handle_runtime_request(reqs)
    return reqs_latency


def start_site_mqtt_agent():
    import socket,os
    client_id = f"{SITE_ID}-{socket.gethostname()}-{os.getpid()}"
    client = mqtt.Client(client_id=client_id, transport="websockets",clean_session=True)
    client.enable_logger()
    # client.tls_set()
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 300)
    initialize_dataloaders()
    print(f"[SiteManager] {SITE_ID} ready. Entering MQTT loop.")
    client.loop_forever()

if __name__ == "__main__":
    start_site_mqtt_agent()
