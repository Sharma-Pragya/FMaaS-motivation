import json, time, threading
import paho.mqtt.client as mqtt
from orchestrator.config import BROKER, PORT, DEPLOYMENT_PLAN_PATH, TIMEOUT
from traces.gamma import generate_requests
import argparse
from collections import defaultdict
from router import route_trace
import os

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
    """Send each site's model + routed runtime requests."""
    num_tasks, alpha, req_rate, cv, duration = config
    print("Publishing deployments + requests to all sites...")

    all_task_names = sorted({
        dec["task"]
        for site in plan["sites"]
        for dep in site["deployments"]
        for dec in dep["decoders"]
    })

    tasks = [(t, None, None) for t in all_task_names]
    trace = generate_requests(num_tasks, alpha, req_rate, cv, duration, tasks, seed)

    routed_trace = route_trace(trace, plan, seed)
    print(routed_trace)
    site_requests = defaultdict(list)
    for r in routed_trace:
        site_requests[r.site_manager].append(r.to_dict())
    
    for site in plan["sites"]:
        site_id = site["id"]
        site_topic = f"fmaas/deploy/site/{site_id}"

        msg = {
            "deployments": site["deployments"],
            "runtime_requests": site_requests.get(site_id, []),
        }

        client.publish(site_topic, json.dumps(msg))
        print(f"[MQTT] Sent deployment + {len(site_requests[site_id])} requests to {site_topic}")
        time.sleep(0.1)
    # save site_requests in csv file 
    # request_latency_results.csv with columns site_manager, device, req_id, req_time
    filename='site_requests.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            f.write('site_manager,device,req_id,req_time\n')
        for site in site_requests:
            for record in site_requests[site]:
                req_id=record['req_id']
                req_time=record['req_time']
                device=record['device']
                f.write(f'{site},{device},{req_id},{req_time}\n')



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
    print(site_ids)
    while time.time() - start < TIMEOUT:
        with acks_lock:
            if all(s in acks for s in site_ids):
                break
        time.sleep(1)
    #save the request latency result from site manager acks in a single csv file not separate per site manager
    #only first time write the header
    #else just append the data
    #get site requests from csv file site_requests.csv
    site_requests = {}
    with open('site_requests.csv', 'r') as f:
        next(f)  # skip header
        for line in f:
            site_manager, device, req_id, req_time = line.strip().split(',')
            if site_manager not in site_requests:
                site_requests[site_manager] = {}
            site_requests[site_manager][int(req_id)] = {
                'device': device,
                'req_time': float(req_time)
            }
    for site in site_ids:
        if site in acks:
            latency_data=acks[site].get('latency',[])
            if latency_data:
                filename='request_latency_results.csv'
                file_exists = os.path.isfile(filename)
                with open(filename, 'a') as f:
                    if not file_exists:
                        f.write('req_id,req_time,site_manager,device,latency\n')
                    for record in latency_data:
                        req_id=record[0]
                        req_time = site_requests[site][req_id]['req_time']
                        site_manager=site
                        device = site_requests[site][req_id]['device']
                        latency=record[1]*1000  # convert to milliseconds
                        f.write(f'{req_id},{req_time},{site_manager},{device},{latency}\n')
                    
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

        config = (1, 1, 6, 1, 10)  # num_tasks, alpha, req_rate, cv, duration
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
