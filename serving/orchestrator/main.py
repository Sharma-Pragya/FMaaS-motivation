import json, time, threading
import paho.mqtt.client as mqtt
from orchestrator.config import BROKER, PORT, DEPLOYMENT_PLAN_PATH, TIMEOUT
import argparse
from collections import defaultdict
from router import route_trace
import os
import ssl
from storage import write_data_to_file
acks = {}
acks_lock = threading.Lock()
all_acks_event = threading.Event()
connected_event = threading.Event()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Orchestrator connected to MQTT broker")
        client.subscribe("fmaas/deploytime/ack/#",qos=1)
        client.subscribe("fmaas/runtime/ack/#", qos=1)
        connected_event.set()
    else:
        print(f"MQTT connection failed (code {rc})")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = json.loads(msg.payload.decode())
    if topic.startswith("fmaas/deploytime/ack"):
        site = payload.get("site", "unknown")
        with acks_lock:
            acks[site] = payload
            if all(s in acks for s in site_ids):
                all_acks_event.set()  # signal done
        print(f"ACK from {site}: {payload}")
            
    elif topic.startswith("fmaas/runtime/ack"):
        print(payload)
        site = payload.get("site", "unknown")
        with acks_lock:
            acks[site] = payload
            if all(s in acks for s in site_ids):
                all_acks_event.set()  # signal done
        print(f"ACK from {site} for runtime: {payload}")

        #not able to send big chunks of data via mqtt hence storing at site_manager
        # site = payload["site"]

        # with acks_lock:
        #     if site not in acks:
        #         acks[site] = {
        #             "total_requests": payload["total_requests"],
        #             "latency": []
        #         }
            
        #     acks[site]["latency"].extend(payload["latency"])

        #     print(acks[site]["total_requests"], len(acks[site]["latency"]))

        #     # === Check completion correctly ===
        #     done = True
        #     for s in site_ids:
        #         if s not in acks:
        #             done = False
        #             break
        #         if len(acks[s]["latency"]) != acks[s]["total_requests"]:
        #             done = False
        #             break
            
        #     if done:
        #         print("All sites complete! Waiting to flush...")
        #         time.sleep(2)
        #         for s in site_ids:
        #             write_data_to_file(s, acks[s])
        #         all_acks_event.set()
            
    
def run_deployment_plan(devices, tasks_slo):
    #old greedy
    # from hueristic.greedy import shared_packing, build_final_json
    # task_manifest=shared_packing(devices,tasks_slo)
    # final_json = build_final_json(task_manifest)
    # with open(f"{DEPLOYMENT_PLAN_PATH}.json", "w") as f:
    #     json.dump(final_json, f, indent=2)
    #new greedy
    from hueristic.greedy_new import shared_packing, build_final_json
    task_manifest = shared_packing(devices,tasks_slo)
    final_json = build_final_json(task_manifest)
    with open(f"{DEPLOYMENT_PLAN_PATH}.json", "w") as f:
        json.dump(final_json, f, indent=2)
    return final_json

def publish_deployments(client, plan, routed_trace):
    """Send each site's model + routed runtime requests."""
    site_requests = defaultdict(list)
    for r in routed_trace:
        site_requests[r.site_manager].append(r.to_dict())
    
    for site in plan["sites"]:
        site_id = site["id"]
        reqs=site_requests.get(site_id, [])
        chunk_length=3000
        i=0
        while i<len(reqs):
            request_msg = {
            "runtime_requests": reqs[i:i+chunk_length],
            }
            print(f"[MQTT] Sent request chunck {i} to {i+chunk_length} {site_id}")
            client.publish(f"fmaas/deploy/site/{site_id}/req", json.dumps(request_msg),qos=1)
            i+=chunk_length
            time.sleep(5)
        deploy_msg = {
            "deployments": site["deployments"],
        }
        client.publish(f"fmaas/deploy/site/{site_id}", json.dumps(deploy_msg),qos=1)
        print(f"[MQTT] Sent deployment to {site_id}")
        time.sleep(0.1)

    #not able to send big chunks of data via mqtt hence storing at site_manager
    # # save site_requests in csv file 
    # # request_latency_results.csv with columns site_manager, device, req_id, req_time
    # filename='site_requests.csv'
    # file_exists = os.path.isfile(filename)
    # with open(filename, 'a') as f:
    #     if not file_exists:
    #         f.write('site_manager,device,backbone,req_id,task,req_time\n')
    #     for site in site_requests:
    #         for record in site_requests[site]:
    #             req_id=record['req_id']
    #             req_time=record['req_time']
    #             device=record['device']
    #             backbone=record['backbone']
    #             task=record['task']
    #             f.write(f'{site},{device},{backbone},{req_id},{task},{req_time}\n')

def trigger_runtime_start(client, plan):
    print("Triggering runtime start on all sites...")
    for site in plan["sites"]:
        topic = f"fmaas/runtime/start/site/{site['id']}"
        client.publish(topic, json.dumps({"command": "start"}),qos=1)
        print(f"Runtime start published to {topic}")
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-only", action="store_true", help="Deployment phase")
    parser.add_argument("--run-only", action="store_true", help="Runtime phase")
    args = parser.parse_args()

    #1. First set the config file for defining tasks and devices
    #extract tasks and devices from config.py
    from experiments.exp5.stage2_50.user_config import devices, tasks 
    all_task_names = sorted({t for t in tasks.keys()})
    routed_tasks = [(t, None, None, None) for t in all_task_names] #task, site, device, backbone
    seed=42

    #2. Connect with the trace generation
    ##synthetic trace generation
    # from traces.gamma import generate_requests
    # num_tasks, alpha, req_rate, cv, duration = (2, 1, 3, 1, 120)  # num_tasks, alpha, req_rate, cv, duration
    # trace,avg_workload_per_task,peak_workload_per_task = generate_requests(num_tasks, alpha, req_rate, cv, duration, routed_tasks, seed)
    # #update tasks dict with peak workload based on real world trace
    # for t in tasks:
    #     if t in avg_workload_per_task:
    #         tasks[t]['peak_workload'] = avg_workload_per_task[t]

    # # # #lmsyschat
    # from traces.lmsyschat import generate_requests
    # req_rate, duration = (200, 360) #max (50,300), (100,300), (150,300), (200,300)
    # trace,avg_workload_per_task,peak_workload_per_task = generate_requests( req_rate,  duration, routed_tasks, seed)
    # #update tasks dict with peak workload based on real world trace
    # for t in tasks:
    #     if t in avg_workload_per_task:
    #         tasks[t]['peak_workload'] = avg_workload_per_task[t]
    # print("Updated tasks:", tasks)

    # #chatbotarena
    # from traces.chatbotarena import generate_requests
    # req_rate, duration = (1,10) #max (50,300), (100,300), (150,300), (200,300)
    # trace,avg_workload_per_task,peak_workload_per_task = generate_requests(req_rate, duration, routed_tasks, seed)
    # #update tasks dict with peak workload based on real world trace
    # for t in tasks:
    #     if t in avg_workload_per_task:
    #         tasks[t]['peak_workload'] = avg_workload_per_task[t]

    #make plan using greedy algorithm and route the trace based on the plan
    plan = run_deployment_plan(devices, tasks)
    # routed_trace = route_trace(trace, plan, seed)
    
    site_ids = [s["id"] for s in plan["sites"]]
    if args.deploy_only:
        acks.clear()
        all_acks_event.clear()
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.enable_logger()
        # client.tls_set()
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start() 
        # WAIT until we are actually subscribed
        if not connected_event.wait(timeout=10):
            raise RuntimeError("MQTT not connected/subscribed in time.")
        print("Publishing deployments + requests to all sites...")
        publish_deployments(client, plan, routed_trace)
        all_acks_event.wait(timeout=TIMEOUT)
        client.disconnect()
        client.loop_stop()        
        print("Deployment phase complete.")

    elif args.run_only:
        acks.clear()
        all_acks_event.clear()
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.enable_logger()
        # client.tls_set()
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        trigger_runtime_start(client, plan)
        all_acks_event.wait(timeout=TIMEOUT)
        client.disconnect()
        client.loop_stop()
        print("Runtime phase complete.")
