import json, asyncio, time
import paho.mqtt.client as mqtt
from site_manager.config import BROKER, PORT, SITE_ID
from site_manager.storage import store_plan_and_requests, get_requests
from site_manager.runtime_executor import handle_runtime_request, initialize_dataloaders
from site_manager.deployment_handler import deploy_models
import ssl
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Site {SITE_ID} connected to broker")
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}")
        client.subscribe(f"fmaas/runtime/start/site/{SITE_ID}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = json.loads(msg.payload.decode())
    if topic.endswith(f"deploy/site/{SITE_ID}"):
        print(f"[MQTT] Received deployment + runtime plan for {SITE_ID}")
        print(payload)
        store_plan_and_requests(payload)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(deploy_models(payload["deployments"]))
        except RuntimeError:
            asyncio.run(deploy_models(payload["deployments"]))
        client.publish(f"fmaas/runtime/ack/site/{SITE_ID}", json.dumps({'site':'site1','deploy':'done'}))
    elif topic.endswith(f"runtime/start/site/{SITE_ID}"):
        print(f"[MQTT] Start signal received for {SITE_ID}")
        asyncio.run(execute_cached_requests(client))

async def execute_cached_requests(client):
    start = time.time()
    reqs = get_requests()
    reqs_latency =await handle_runtime_request(reqs)
    ack = {
        "site": SITE_ID,
        "status": "completed",
        "total_requests": len(reqs),
        "latency": reqs_latency,
        "runtime_duration": time.time() - start,
    }
    client.publish(f"fmaas/runtime/ack/site/{SITE_ID}", json.dumps(ack))
    print(f"[MQTT] Sent runtime ACK for {SITE_ID}")

def start_site_mqtt_agent():
    client = mqtt.Client(client_id=SITE_ID, transport="websockets")
    # client.tls_set()
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    initialize_dataloaders()
    client.loop_forever()

if __name__ == "__main__":
    start_site_mqtt_agent()
