import time

import torch
from transformers import pipeline, AutoTokenizer


def count_tokens(messages, tokenizer):
    # Combine all message contents to calculate token length
    combined_text = " ".join([message["content"] for message in messages])
    tokens = tokenizer(combined_text)
    return len(tokens['input_ids'])


if __name__ == '__main__':
    # messages = [
    #     {"role": "user", "content": "Who are you?"},
    # ]
    # pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
    # # pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
    #
    # pipe(messages)

    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    # Start timing the pipeline setup
    start_time = time.time()
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Measure the time taken to load the pipeline
    setup_time = time.time() - start_time
    print(f"Pipeline setup time: {setup_time:.2f} seconds")

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Count tokens in the input messages
    token_count = count_tokens(messages, tokenizer)
    print(f"Number of tokens in the input messages: {token_count}")

    start_time = time.time()
    outputs = pipe(
        messages,
        max_new_tokens=1,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")
    print(outputs[0]["generated_text"][-1])
