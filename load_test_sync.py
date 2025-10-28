import asyncio
import aiohttp
import time
import random

# 🔹 Replace with your Hugging Face Space URL
ORCHESTRATOR_URL = "http://127.0.0.1:8000/ticket"
# If running locally:
# ORCHESTRATOR_URL = "http://127.0.0.1:8000/ticket"

# 25 random payloads to simulate user tickets
payloads = [
    {
        "channel": "web",
        "severity": random.choice(["high", "low"]),
        "summary": f"Test orchestrator ticket #{i}"
    }
    for i in range(25)
]

async def send_request(session, idx, payload):
    start = time.time()
    try:
        async with session.post(ORCHESTRATOR_URL, json=payload) as resp:
            elapsed = time.time() - start
            print(f"#{idx} → Status: {resp.status}, Time: {elapsed:.2f}s")
            return resp.status, elapsed
    except Exception as e:
        print(f"#{idx} → Failed: {e}")
        return None, 0

async def main():
    print("🚀 Sending 25 concurrent requests to ORCHESTRATOR...\n")
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i, payloads[i]) for i in range(len(payloads))]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    success = sum(1 for s, _ in results if s == 200)
    avg_time = sum(t for _, t in results if t > 0) / len(results)

    print(f"\n✅ Done! Success: {success}/{len(results)} | Avg Time: {avg_time:.2f}s | Total Duration: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
