import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Use torch.multinomial() to choose the next token.
        #    This function simulates a weighted draw from a given list of probabilities
        #    It's similar to picking marbles out of a bag.
        # 2. the given model's output is BEFORE softmax is applied,
        #    and the forward() output has shape batch X time X vocab_size
        # 3. Do not alter the code below, only add to it. This is for maintaining reproducibility in testing.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        res = []
        print(context.shape)
        print(context.T)
        for i in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            prediction = model(context)  # B, T, Vocab_Size
            last_time_step = prediction[:, -1, :]  # B, Vocab_Size
            probabilities = nn.functional.softmax(last_time_step, dim=-1)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            next_char = torch.multinomial(probabilities, 1, generator=generator)

            generator.set_state(initial_state)
            # MORE OF YOUR CODE (arbitrary number of lines)
            context = torch.cat((context, next_char), dim=-1)
            res.append(int_to_char[next_char.item()])

        # Once your code passes the test, check out the Colab link and hit Run to see your code generate new Drake lyrics!
        # Your code's output, ran in this sandbox will be boring because of the computational limits in this sandbox
        return ''.join(res)
