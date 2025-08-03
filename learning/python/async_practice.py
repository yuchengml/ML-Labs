import time
import asyncio

def synchronous_execution():
    """
    Demonstrates synchronous execution.
    """
    def count_sync():
        print("Sync: One")
        time.sleep(1)
        print("Sync: Two")

    def main_sync():
        for _ in range(3):
            count_sync()

    print("\n--- Synchronous Execution ---")
    start = time.perf_counter()
    main_sync()
    elapsed = time.perf_counter() - start
    print(f"同步執行耗時：{elapsed:.2f} 秒")

async def asynchronous_execution():
    """
    Demonstrates asynchronous execution using asyncio.
    """
    async def count_async():
        print("Async: One")
        await asyncio.sleep(1)
        print("Async: Two")

    async def main_async():
        await asyncio.gather(count_async(), count_async(), count_async())

    print("\n--- Asynchronous Execution ---")
    start = time.perf_counter()
    await main_async()
    elapsed = time.perf_counter() - start
    print(f"非同步執行耗時：{elapsed:.2f} 秒")

if __name__ == "__main__":
    synchronous_execution()
    asyncio.run(asynchronous_execution())