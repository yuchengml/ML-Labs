import multiprocessing
import subprocess
import time

import requests


def serve_vllm():
    # Set any required environment variables here
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Example: use GPU 0

    # Command to start the vLLM server
    # cmd = [
    #     "python3",
    #     "-m",
    #     "vllm.entrypoints.api_server",
    #     "--model",
    #     "meta-llama_Llama-3.2-1B-Instruct",  # Replace with your model
    #     "--port",
    #     "8000"  # Specify the desired port
    # ]
    cmd = [
        "uvicorn",
        "simple_api:app",  # Replace 'app' with your FastAPI app module name
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ]

    # Start the vLLM server
    subprocess.run(cmd)


def call_api_periodically():
    # Wait for the server to start
    time.sleep(10)
    for i in range(5):
        print(f"This is the {i + 1}/5 times API calling")
        try:
            response = requests.get("http://127.0.0.1:8000/")
            print(f"API Call {i + 1}: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"API call {i + 1} failed: {e}")
        time.sleep(5)


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method("spawn")

    # Create and start the child process
    p = multiprocessing.Process(target=serve_vllm)
    p.start()

    # Main process continues without waiting for the child
    print("Main process continues...")

    # Call the API periodically in the main process
    call_api_periodically()

    # Terminate the child process after 10 API calls
    p.terminate()
    p.join()
    print("FastAPI server has been terminated.")
