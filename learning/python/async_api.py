from fastapi import FastAPI
import time
import asyncio
import uvicorn  # Import uvicorn
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get("/sync")
def sync_endpoint():
    time.sleep(2)  # Simulate a time-consuming operation
    return {"message": "Synchronous task finished"}


@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(2)  # Asynchronous wait, does not block the main program
    return {"message": "Asynchronous task finished"}


def generate_progress():
    for i in range(101):
        yield f"progress:{i}\n"
        time.sleep(0.1)  # 模擬運算


@app.get("/stream-progress")
async def stream_progress():
    return StreamingResponse(generate_progress(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
