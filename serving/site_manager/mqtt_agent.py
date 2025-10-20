import json
import asyncio
import paho.mqtt.client as mqtt
from site_manager.deployment_handler import deploy_models, DeploySpec

# Public broker (works over HTTPS/WebSocket)
BROKER = "broker.emqx.io"
PORT = 8084   
SITE_ID = "site1"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Site {SITE_ID} connected to broker {BROKER}:{PORT}")
        client.subscribe(f"fmaas/deploy/site/{SITE_ID}")
        print(f"Subscribed to fmaas/deploy/site/{SITE_ID}")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        print(f" Deployment command received: {msg.payload.decode()}")
        specs = json.loads(msg.payload.decode())
        asyncio.run(deploy_models(specs))
        ack = {"status": "loaded", "site": SITE_ID, "num_models": len(specs)}
        client.publish(f"fmaas/ack/site/{SITE_ID}", json.dumps(ack))
        print("Sent ACK to orchestrator")
    except Exception as e:
        print(f"Error in handling deployment: {e}")

def start_site_mqtt():
    client = mqtt.Client(client_id=SITE_ID, transport="websockets")
    client.tls_set()  # enable TLS
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    start_site_mqtt()
