# site_manager/health_monitor.py
import asyncio, aiohttp, time

async def heartbeat(orchestrator_url: str, site_id: str):
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{orchestrator_url}/heartbeat", json={"site_id": site_id, "timestamp": time.time()})
        except Exception as e:
            print(f"[HealthMonitor] Heartbeat failed: {e}")
        await asyncio.sleep(10)
