from llama_cpp import Llama

# conda create -n llama-cpp-python python=3.9
# pip install llama-cpp-python huggingface-hub


# llm = Llama.from_pretrained(
# 	repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
# 	filename="./prebuilt_model/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
# )

llm = Llama(
    model_path="./prebuilt_model/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"
)

result = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
)

print(result)
