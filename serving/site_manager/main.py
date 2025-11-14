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
            loop.create_task(deploy_models(payload["deployments"]))
        except RuntimeError:
            asyncio.run(deploy_models(payload["deployments"]))
        client.publish(f"fmaas/deploytime/ack/site/{SITE_ID}", json.dumps({'site':SITE_ID,'deploy':'done'}), qos=1)
        print(f"[MQTT] Sent deploytime ACK for {SITE_ID}")

    elif topic.endswith(f"deploy/site/{SITE_ID}/req"):
        print(f"[MQTT] Received runtime plan for {SITE_ID}")
        store_requests(payload)

    elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
        print(f"[MQTT] Start signal received for {SITE_ID}")
        reqs_latency=asyncio.run(execute_cached_requests(client))
        ## send small chuncks of data
        chunk_length=1500
        i=0        
        while i<len(reqs_latency):
            print(f"[MQTT] Sent request chunck {i} to {i+chunk_length}")
            payload = {
                "site": SITE_ID,
                "status": "completed",
                "total_requests": len(reqs_latency),
                "latency": reqs_latency[i:i+chunk_length],
            }
            # payload_json = json.dumps(payload)
            # payload_bytes = payload_json.encode("utf-8")
            # size_bytes = len(payload_bytes)

            # print(f"Payload size: {size_bytes} bytes (~{size_bytes/1024:.2f} KB)")
            client.publish(f"fmaas/runtime/ack/site/{SITE_ID}", json.dumps(payload),qos=1)
            i+=chunk_length
            time.sleep(5)
        print(f"[MQTT] Sent runtime ACK for {SITE_ID}")

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
