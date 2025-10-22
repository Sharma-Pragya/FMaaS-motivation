import json, time, threading
import paho.mqtt.client as mqtt
from orchestrator.config import BROKER, PORT, DEPLOYMENT_PLAN_PATH, TIMEOUT
from traces.gamma import generate_requests
from orchestrator.router import get_route
import argparse

acks = {}
acks_lock = threading.Lock()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Orchestrator connected to MQTT broker")
        client.subscribe("fmaas/runtime/ack/#")
    else:
        print(f"MQTT connection failed (code {rc})")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    site = payload.get("site", "unknown")
    with acks_lock:
        acks[site] = payload
    print(f"ACK from {site}: {payload}")

def load_deployment_plan():
    with open(DEPLOYMENT_PLAN_PATH, "r") as f:
        return json.load(f)

def publish_deployments(client, plan, config, seed):
    """Send each site's model + runtime requests."""
    num_tasks, alpha, req_rate, cv, duration = config
    print("Publishing deployments + requests to all sites...")
    for site in plan["sites"]:
        site_id = site["id"]
        site_topic = f"fmaas/deploy/site/{site_id}"

        ## need to work on this make common workload 
        ## and send the appropriate request to the deployment
        tasks = []
        for task_name in ["hr"]:
        # for task_name in ["vqa"]:
            site_manager, device = get_route(task_name)
            tasks.append((task_name, site_manager, device))
        requests = generate_requests(num_tasks, alpha, req_rate, cv, duration, tasks, seed)

        msg = {
            "deployments": site["deployments"],
            "runtime_requests": [r.to_dict() for r in requests],
        }
        client.publish(site_topic, json.dumps(msg))
        print(f"Sent deployment to {site_topic}")
        time.sleep(0.1)

def trigger_runtime_start(client, plan):
    print("Triggering runtime start on all sites...")
    for site in plan["sites"]:
        topic = f"fmaas/runtime/start/site/{site['id']}"
        client.publish(topic, json.dumps({"command": "start"}))
        print(f"Runtime start published to {topic}")
        time.sleep(0.05)

def wait_for_acks(site_ids):
    print(f"Waiting up to {TIMEOUT}s for site ACKs...")
    start = time.time()
    while time.time() - start < TIMEOUT:
        with acks_lock:
            if all(s in acks for s in site_ids):
                break
        time.sleep(1)
    print("\nSummary:")
    for sid in site_ids:
        if sid in acks:
            print(f"{sid}: {acks[sid]}")
        else:
            print(f"No ACK from {sid} (timeout)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-only", action="store_true", help="Deployment phase")
    parser.add_argument("--run-only", action="store_true", help="Runtime phase")
    args = parser.parse_args()
    plan = load_deployment_plan()
    site_ids = [s["id"] for s in plan["sites"]]
    if args.deploy_only:
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.tls_set()
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start()

        config = (1, 1, 6, 1, 1)
        seed=42
        publish_deployments(client, plan, config, seed)
        wait_for_acks(site_ids)
        client.loop_stop()
        client.disconnect()
        print("Deployment phase complete.")

    elif args.run_only:
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.tls_set()
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start()

        trigger_runtime_start(client, plan)
        wait_for_acks(site_ids)

        client.loop_stop()
        client.disconnect()
        print("Runtime phase complete.")
