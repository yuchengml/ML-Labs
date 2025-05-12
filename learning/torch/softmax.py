import torch


class Softmax(torch.nn.Module):
    """
    Compute softmax values for each set of logits in input tensor.

    Arguments:
    logits: Tensor of shape (batch_size, num_classes) containing raw scores/logits.

    Returns:
    Tensor of shape (batch_size, num_classes) containing softmax probabilities.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        # Compute exponential of logits
        exp_logits = torch.exp(logits)

        # Compute sum of exponentials along the classes dimension
        sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)

        # Compute softmax probabilities by dividing exp(logits) by sum(exp(logits))
        softmax_probs = exp_logits / sum_exp_logits

        return softmax_probs


if __name__ == '__main__':
    # Example usage:
    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.0, 0.3]])

    softmax = Softmax()
    softmax_probs = softmax(logits)
    print("Softmax probabilities:", softmax_probs)
