import json, asyncio, time, gzip
import paho.mqtt.client as mqtt
from site_manager.config import BROKER, PORT, SITE_ID
from site_manager.storage import store_plan, store_requests, get_requests
from site_manager.runtime_executor import handle_runtime_request, initialize_dataloaders
from site_manager.deployment_handler import deploy_models
import ssl
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Site {SITE_ID} connected to broker")
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}",qos=1)
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}/req",qos=1)
        client.subscribe(f"fmaas/runtime/start/site/{SITE_ID}",qos=1)
    else:
        print(f"[MQTT] Connection failed with code {rc}")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = json.loads(msg.payload.decode())
    if topic.endswith(f"deploy/site/{SITE_ID}"):
        print(f"[MQTT] Received deployment + runtime plan for {SITE_ID}")
        store_plan(payload)
        try:
            loop = asyncio.get_running_loop()
            deployment_status = loop.create_task(deploy_models(payload["deployments"]))
        except RuntimeError:
            deployment_status = asyncio.run(deploy_models(payload["deployments"]))
        #save deployment_status in json file model_deployment_results.json
        with open("model_deployment_results.json","w") as f:
            json.dump(deployment_status,f,indent=4)
        
        client.publish(f"fmaas/deploytime/ack/site/{SITE_ID}", json.dumps({'site':SITE_ID,'deploymentstatus':deployment_status}), qos=1)
        print(f"[MQTT] Sent deploytime ACK for {SITE_ID}")

    elif topic.endswith(f"deploy/site/{SITE_ID}/req"):
        print(f"[MQTT] Received runtime plan for {SITE_ID}")
        store_requests(payload)

    elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
        print(f"[MQTT] Start signal received for {SITE_ID}")
        reqs_latency=asyncio.run(execute_cached_requests(client))
        #save reqs_latency to file request_latency_results.csv
        # req_id,req_time,site_manager,device,backbone,task,end_to_end_latency(ms),proc_time(ms),swap_time(ms),pred,true
        # out of req_id,end_to_end_latency(ms),proc_time(ms),swap_time(ms),pred,true get from reqs_latency
        #rest from stored requests
        reqs = get_requests()
        with open(f"request_latency_results.csv","w") as f:
            f.write("req_id,req_time,site_manager,device,backbone,task,end_to_end_latency(ms),proc_time(ms),swap_time(ms),pred,true\n")
            reqs_dict = {req['req_id']:req for req in reqs}
            for entry in reqs_latency:
                req_id,device_url,e2e_latency,proc_time,swap_time,pred,true = entry
                req = reqs_dict.get(req_id,{})
                req_time = req.get('req_time',-1)
                site_manager = SITE_ID
                device = device_url
                backbone = req.get('backbone','unknown')
                task = req.get('task','unknown')
                f.write(f"{req_id},{req_time},{site_manager},{device},{backbone},{task},{e2e_latency*1000:.2f},{proc_time*1000:.2f},{swap_time*1000:.2f},{pred},{true}\n")
        payload = {
            "site": SITE_ID,
            "status": "completed",
            "total_requests": len(reqs_latency),
        }
        client.publish(f"fmaas/runtime/ack/site/{SITE_ID}", json.dumps(payload),qos=1)
        print(f"[MQTT] Sent runtime ACK for {SITE_ID}")

        #not able to send big chunks of data via mqtt hence storing at site_manager
        ## send small chuncks of data
        # chunk_length=1500
        # i=0        
        # while i<len(reqs_latency):
        #     print(f"[MQTT] Sent request chunck {i} to {i+chunk_length}")
        #     payload = {
        #         "site": SITE_ID,
        #         "status": "completed",
        #         "total_requests": len(reqs_latency),
        #         "latency": reqs_latency[i:i+chunk_length],
        #     }
        #     # payload_json = json.dumps(payload)
        #     # payload_bytes = payload_json.encode("utf-8")
        #     # size_bytes = len(payload_bytes)
        #     # print(f"Payload size: {size_bytes} bytes (~{size_bytes/1024:.2f} KB)")
        #     client.publish(f"fmaas/runtime/ack/site/{SITE_ID}", json.dumps(payload),qos=1)
        #     i+=chunk_length
        #     time.sleep(5)
        # print(f"[MQTT] Sent runtime ACK for {SITE_ID}")

async def execute_cached_requests(client):
    start = time.time()
    reqs = get_requests()
    reqs_latency =await handle_runtime_request(reqs)
    return reqs_latency


def start_site_mqtt_agent():
    client = mqtt.Client(client_id=SITE_ID, transport="websockets",clean_session=True)
    client.enable_logger()
    # client.tls_set()
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 300)
    initialize_dataloaders()
    client.loop_forever()

if __name__ == "__main__":
    start_site_mqtt_agent()
