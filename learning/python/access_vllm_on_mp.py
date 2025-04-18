import multiprocessing
import subprocess
import os
import time

def serve_vllm():
    # Set any required environment variables here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Example: use GPU 0

    # Command to start the vLLM server
    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.api_server",
        "--model",
        "meta-llama_Llama-3.2-1B-Instruct",  # Replace with your model
        "--port",
        "8000"  # Specify the desired port
    ]

    # Start the vLLM server
    subprocess.run(cmd)

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method("spawn")

    # Create and start the child process
    p = multiprocessing.Process(target=serve_vllm)
    p.start()

    # Main process continues without waiting for the child
    print("Main process continues...")
    # Optionally do more work here...
    time.sleep(30)  # give child time to print before main exits
