import requests
import time
import aiohttp
import asyncio


def call_sync_api():
    """
    Calls a synchronous API three times and measures the total time taken.
    """
    start = time.perf_counter()
    for _ in range(3):
        response = requests.get("http://localhost:8000/sync")
        print(response.json())
    elapsed = time.perf_counter() - start
    print(f"Total time for synchronous calls: {elapsed:.2f} seconds")


async def call_async_api():
    """
    Calls an asynchronous API three times concurrently and measures the total time taken.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [session.get("http://localhost:8000/async") for _ in range(3)]
        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        for r in responses:
            print(await r.json())
        elapsed = time.perf_counter() - start
        print(f"Total time for asynchronous calls: {elapsed:.2f} seconds")


async def call_stream_api():
    """
    Calls the streaming API and processes the streamed response.
    """
    print("\nCalling streaming API...")
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/stream-progress") as response:
            print(f"Status: {response.status}")
            async for chunk in response.content.iter_any():
                # Decode the chunk and print it. Assuming UTF-8 encoding.
                print(f"Received: {chunk.decode('utf-8').strip()}")
    print("Streaming API call finished.")


if __name__ == "__main__":
    # Execute synchronous API calls
    call_sync_api()

    # Execute asynchronous API calls
    asyncio.run(call_async_api())

    # Execute streaming API call
    asyncio.run(call_stream_api())
