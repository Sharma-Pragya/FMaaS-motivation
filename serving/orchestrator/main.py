import json, time, threading
import paho.mqtt.client as mqtt
from orchestrator.config import BROKER, PORT, DEPLOYMENT_PLAN_PATH, TIMEOUT
import argparse
from collections import defaultdict
from router import route_trace
import os
from hueristic.greedy import shared_packing, build_final_json

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

def run_deployment_plan(devices, tasks_slo):
    task_manifest=shared_packing(devices,tasks_slo)
    final_json = build_final_json(task_manifest)
    with open(DEPLOYMENT_PLAN_PATH, "w") as f:
        json.dump(final_json, f, indent=2)
    return final_json

def publish_deployments(client, plan, routed_trace):
    """Send each site's model + routed runtime requests."""

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
            f.write('site_manager,device,backbone,req_id,task,req_time,\n')
        for site in site_requests:
            for record in site_requests[site]:
                req_id=record['req_id']
                req_time=record['req_time']
                device=record['device']
                backbone=record['backbone']
                task=record['task']
                f.write(f'{site},{device},{backbone},{req_id},{task},{req_time}\n')



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
            site_manager, device, backbone, req_id, task, req_time = line.strip().split(',')
            if site_manager not in site_requests:
                site_requests[site_manager] = {}
            site_requests[site_manager][int(req_id)] = {
                'device': device,
                'backbone': backbone,
                'task':task,
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
                        f.write('req_id,req_time,site_manager,device,backbone,task,latency\n')
                    for record in latency_data:
                        req_id=record[0]
                        req_time = site_requests[site][req_id]['req_time']
                        site_manager=site
                        device = site_requests[site][req_id]['device']
                        backbone=site_requests[site][req_id]['backbone']
                        task= site_requests[site][req_id]['task']
                        latency=record[1]*1000  # convert to milliseconds
                        f.write(f'{req_id},{req_time},{site_manager},{device},{backbone},{task},{latency}\n')
                    
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

    #extract tasks and devices from config.py
    from user_config import devices, tasks 
    print("Publishing deployments + requests to all sites...")
    all_task_names = sorted({t for t in tasks.keys()})
    routed_tasks = [(t, None, None, None) for t in all_task_names] #task, site, device, backbone
    seed=42

    # #synthetic trace generation
    # from traces.gamma import generate_requests
    # num_tasks, alpha, req_rate, cv, duration = (1, 1, 6, 1, 10)  # num_tasks, alpha, req_rate, cv, duration
    # trace = generate_requests(num_tasks, alpha, req_rate, cv, duration, routed_tasks, seed)

    #real workload trace can be loaded
    #update the peak workload in tasks based on trace
    # from traces.lmsyschat import generate_requests
    # req_rate, duration = (6, 10)
    # trace,avg_workload_per_task,peak_workload_per_task = generate_requests( req_rate,  duration, routed_tasks, seed)
    # #update tasks dict with peak workload based on real world trace
    # for t in tasks:
    #     if t in peak_workload_per_task:
    #         tasks[t]['peak_workload'] = peak_workload_per_task[t]
    # print("Updated tasks:", tasks)

    from traces.chatbotarena import generate_requests
    req_rate, duration = (10, 10) #max (200,300)
    trace,avg_workload_per_task,peak_workload_per_task = generate_requests(req_rate, duration, routed_tasks, seed)

    #update tasks dict with peak workload based on real world trace
    for t in tasks:
        if t in avg_workload_per_task:
            tasks[t]['peak_workload'] = avg_workload_per_task[t]

    #make plan using greedy algorithm and route the trace based on the plan
    plan = run_deployment_plan(devices, tasks)
    routed_trace = route_trace(trace, plan, seed)

    site_ids = [s["id"] for s in plan["sites"]]
    if args.deploy_only:
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.tls_set()
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        publish_deployments(client, plan, routed_trace)
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
