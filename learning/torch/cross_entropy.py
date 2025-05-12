import torch


class CrossEntropy(torch.nn.Module):
    """
    Computes the cross-entropy loss between y_true and y_pred.

    Arguments:
    y_true: Tensor of true probabilities. Expect a probability distribution like softmax output.
    y_pred: Tensor of predicted probabilities. Expect a probability distribution like softmax output.

    Returns:
    Cross-entropy loss as a scalar tensor.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-10  # Small value to avoid numerical instability

    def forward(self, y_true, y_pred):
        # Clip predicted probabilities to avoid log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)

        loss = -torch.sum(y_true * torch.log(y_pred), dim=-1)  # Calculate cross-entropy loss

        return torch.mean(loss)


if __name__ == '__main__':
    # Generating some example data
    y_true_probs = torch.tensor([[0.2, 0.3, 0.5],  # True probabilities for each class (example 1)
                                 [0.6, 0.3, 0.1]])  # True probabilities for each class (example 2)
    y_pred_probs = torch.tensor([[0.1, 0.6, 0.3],  # Predicted probabilities for each class (example 1)
                                 [0.5, 0.2, 0.3]])  # Predicted probabilities for each class (example 2)
    # Calculate cross-entropy loss
    loss_fn = CrossEntropy()
    loss = loss_fn(y_true_probs, y_pred_probs)
    print("Cross-Entropy Loss:", loss.item())
