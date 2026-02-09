import json, time, threading
import paho.mqtt.client as mqtt
from orchestrator.config import BROKER, PORT, DEPLOYMENT_PLAN_PATH, TIMEOUT
import argparse
from collections import defaultdict
from router import route_trace
import os
import ssl
from orchestrator.storage import write_data_to_file
acks = {}
acks_lock = threading.Lock()
all_acks_event = threading.Event()
connected_event = threading.Event()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Orchestrator connected to MQTT broker")
        client.subscribe("fmaas/deploytime/ack/#", qos=1)
        client.subscribe("fmaas/runtime/ack/#", qos=1)
        client.subscribe("fmaas/cleanup/ack/#", qos=1)
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
        site = payload.get("site", "unknown")
        with acks_lock:
            acks[site] = payload
            if all(s in acks for s in site_ids):
                all_acks_event.set()
        print(f"ACK from {site} for runtime: {payload}")

    elif topic.startswith("fmaas/cleanup/ack"):
        site = payload.get("site", "unknown")
        with acks_lock:
            acks[site] = payload
            if all(s in acks for s in site_ids):
                all_acks_event.set()
        print(f"Cleanup ACK from {site}: {payload}")
            
    
def run_deployment_plan(devices, tasks_slo, scheduler_name='fmaas', output_dir=None):
    """Run deployment planning with the specified scheduler.
    
    Args:
        devices: Device configuration dict.
        tasks_slo: Task specification dict with SLO info.
        scheduler_name: One of 'fmaas', 'clipper', 'm4'.
        output_dir: Directory to save deployment_plan.json. If None, uses default path.
    
    Returns:
        Deployment plan as JSON dict.
    """
    from planner import FMaaSScheduler, ClipperScheduler, M4Scheduler, ProfileData, SchedulerConfig
    from planner.schedulers import fmaas as fmaas_mod
    from planner.schedulers import clipper as clipper_mod
    from planner.schedulers import m4 as m4_mod
    from planner.parser.profiler import components, pipelines, latency, metric
    
    # Create profile data and config
    profile = ProfileData(components, pipelines, latency, metric)
    config = SchedulerConfig(util_factor=0.8)
    
    # Dispatch to the requested scheduler
    if scheduler_name == 'fmaas':
        scheduler = FMaaSScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, share_mode=False)
        final_json = fmaas_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'fmaas_share':
        scheduler = FMaaSScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, share_mode=True)
        final_json = fmaas_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'clipper-ht':
        scheduler = ClipperScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=False)
        final_json = clipper_mod.build_final_json(deployments, pipelines)
    elif scheduler_name=='clipper-ha':
        scheduler = ClipperScheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=True)
        final_json = clipper_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'm4-ht':
        scheduler = M4Scheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=False)
        final_json = m4_mod.build_final_json(deployments, pipelines)
    elif scheduler_name == 'm4-ha':
        scheduler = M4Scheduler(profile, config)
        deployments = scheduler.schedule(devices, tasks_slo, accuracy_mode=True)
        final_json = m4_mod.build_final_json(deployments, pipelines)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Use: fmaas, fmaas_share, clipper, m4")
    
    # Save deployment plan
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plan_path = os.path.join(output_dir, "deployment_plan.json")
    else:
        plan_path = f"{DEPLOYMENT_PLAN_PATH}.json"
    
    with open(plan_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"[Orchestrator] Saved deployment plan to {plan_path} (scheduler={scheduler_name})")
    
    return final_json

def publish_deployments(client, plan, routed_trace, output_dir=None):
    """Send each site's model + routed runtime requests.
    
    The output_dir is included in the deployment message so the site manager
    can save results (CSVs, JSONs) directly into the experiment directory.
    """
    site_requests = defaultdict(list)
    for r in routed_trace:
        site_requests[r.site_manager].append(r.to_dict())
    
    for site in plan["sites"]:
        site_id = site["id"]
        reqs = site_requests.get(site_id, [])

        # 1. Send deploy message FIRST — triggers clear_state() on site manager,
        #    wiping old requests and setting the new output_dir.
        deploy_msg = {
            "deployments": site["deployments"],
        }
        if output_dir:
            deploy_msg["output_dir"] = output_dir
        client.publish(f"fmaas/deploy/site/{site_id}", json.dumps(deploy_msg), qos=1)
        print(f"[MQTT] Sent deployment to {site_id} (output_dir={output_dir})")
        time.sleep(2)  # give site manager time to process deploy before requests

        # 2. Send request chunks AFTER deploy — they accumulate into the
        #    now-clean RUNTIME_REQUESTS buffer.
        chunk_length = 3000
        i = 0
        while i < len(reqs):
            request_msg = {
                "runtime_requests": reqs[i:i+chunk_length],
            }
            print(f"[MQTT] Sent request chunk {i} to {i+chunk_length} {site_id}")
            client.publish(f"fmaas/deploy/site/{site_id}/req", json.dumps(request_msg), qos=1)
            i += chunk_length
            time.sleep(5)
        print(f"[MQTT] All {len(reqs)} requests sent to {site_id}")
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
        client.publish(topic, json.dumps({"command": "start"}), qos=1)
        print(f"Runtime start published to {topic}")
        time.sleep(0.05)


def trigger_cleanup(client, plan):
    """Send cleanup signal to all site managers.
    
    Each site manager will SSH into its devices and kill Triton server
    processes, then ACK back on fmaas/cleanup/ack/site/<id>.
    """
    print("Triggering cleanup on all sites...")
    for site in plan["sites"]:
        topic = f"fmaas/cleanup/site/{site['id']}"
        client.publish(topic, json.dumps({"command": "cleanup"}), qos=1)
        print(f"Cleanup signal published to {topic}")
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FMaaS Orchestrator")
    parser.add_argument("--deploy-only", action="store_true", help="Deployment phase")
    parser.add_argument("--run-only", action="store_true", help="Runtime phase")
    parser.add_argument("--cleanup-only", action="store_true",
                        help="Kill Triton servers on all devices (no deploy/run)")
    parser.add_argument("--scheduler", type=str, default="fmaas",
                        choices=["fmaas", "fmaas_share", "clipper-ht", "m4-ht", "clipper-ha", "m4-ha"],
                        help="Scheduler to use: fmaas, fmaas_share, clipper-ht, m4-ht, clipper-ha, m4-ha")
    parser.add_argument("--req-rate", type=int, default=10,
                        help="Request rate (reqs/sec) for trace generation")
    parser.add_argument("--duration", type=int, default=360,
                        help="Trace duration in seconds")
    parser.add_argument("--exp-dir", type=str, default=".",
                        help="Base experiment output directory")
    parser.add_argument("--trace", type=str, default="lmsyschat",
                        choices=["lmsyschat", "gamma", "chatbotarena"],
                        help="Trace generator to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Output directory: <exp_dir>/<scheduler>/<req_rate>/ (always absolute) ---
    output_dir = os.path.abspath(os.path.join(args.exp_dir, args.scheduler, str(args.req_rate)))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load task/device config
    from experiments.baselines.user_config import devices, tasks
    all_task_names = sorted({t for t in tasks.keys()})
    routed_tasks = [(t, None, None, None) for t in all_task_names]  # task, site, device, backbone

    # 2. Generate trace
    if args.trace == 'lmsyschat':
        from traces.lmsyschat import generate_requests
        trace, avg_workload_per_task, peak_workload_per_task = generate_requests(
            args.req_rate, args.duration, routed_tasks, args.seed
        )
    elif args.trace == 'gamma':
        from traces.gamma import generate_requests
        num_tasks, alpha, cv = len(all_task_names), 1, 1
        trace, avg_workload_per_task, peak_workload_per_task = generate_requests(
            num_tasks, alpha, args.req_rate, cv, args.duration, routed_tasks, args.seed
        )
    elif args.trace == 'chatbotarena':
        from traces.chatbotarena import generate_requests
        trace, avg_workload_per_task, peak_workload_per_task = generate_requests(
            args.req_rate, args.duration, routed_tasks, args.seed
        )

    # Update tasks dict with workload from trace
    for t in tasks:
        if t in avg_workload_per_task:
            tasks[t]['peak_workload'] = avg_workload_per_task[t]
    print(f"[Orchestrator] scheduler={args.scheduler}, req_rate={args.req_rate}, "
          f"trace={args.trace}, output={output_dir}")
    print(f"[Orchestrator] Updated tasks: {tasks}")

    # 3. Run planner and route trace
    plan = run_deployment_plan(devices, tasks, scheduler_name=args.scheduler, output_dir=output_dir)
    routed_trace = route_trace(trace, plan, args.seed)
    site_ids = [s["id"] for s in plan["sites"]]

    if args.deploy_only:
        acks.clear()
        all_acks_event.clear()
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.enable_logger()
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
        publish_deployments(client, plan, routed_trace, output_dir=output_dir)
        all_acks_event.wait(timeout=TIMEOUT)
        client.disconnect()
        client.loop_stop()
        print(f"Deployment phase complete. Results in {output_dir}")

    elif args.run_only:
        acks.clear()
        all_acks_event.clear()
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.enable_logger()
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        if not connected_event.wait(timeout=10):
            raise RuntimeError("MQTT not connected/subscribed in time.")

        # 1. Trigger runtime
        trigger_runtime_start(client, plan)
        all_acks_event.wait(timeout=TIMEOUT)
        print(f"Runtime phase complete. Results in {output_dir}")

        # 2. Trigger cleanup (kill Triton servers)
        acks.clear()
        all_acks_event.clear()
        trigger_cleanup(client, plan)
        all_acks_event.wait(timeout=120)  # cleanup should be fast
        print("Device cleanup complete.")

        client.disconnect()
        client.loop_stop()

    elif args.cleanup_only:
        # Standalone cleanup: kill all Triton servers without running experiments
        acks.clear()
        all_acks_event.clear()
        client = mqtt.Client(client_id="orchestrator", transport="websockets")
        client.enable_logger()
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_message = on_message
        print(f"Connecting to {BROKER}:{PORT} ...")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        if not connected_event.wait(timeout=10):
            raise RuntimeError("MQTT not connected/subscribed in time.")
        trigger_cleanup(client, plan)
        all_acks_event.wait(timeout=120)
        client.disconnect()
        client.loop_stop()
        print("Cleanup-only complete. All device servers killed.")

    else:
        # Plan-only mode: just generate and save the deployment plan + routed trace info
        print(f"[Orchestrator] Plan-only mode. Deployment plan saved to {output_dir}")
        # Save trace summary for reference
        trace_summary = {
            "scheduler": args.scheduler,
            "req_rate": args.req_rate,
            "duration": args.duration,
            "trace": args.trace,
            "seed": args.seed,
            "avg_workload_per_task": avg_workload_per_task,
            "peak_workload_per_task": peak_workload_per_task,
            "num_requests": len(trace),
            "num_routed": len(routed_trace),
        }
        with open(os.path.join(output_dir, "trace_summary.json"), "w") as f:
            json.dump(trace_summary, f, indent=2)
