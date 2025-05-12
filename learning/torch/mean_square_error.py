import torch
import torch.nn as nn


class MeanSquareError(nn.Module):
    """
    Computes the mean square error between y_true and y_pred.

    Arguments:
    y_true: Tensor of true values.
    y_pred: Tensor of predicted values.

    Returns:
    Mean square error as a scalar tensor.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


if __name__ == '__main__':
    y_true = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y_pred = torch.tensor([2, 2.5, 3.5, 4.5], dtype=torch.float32)
    mse_error = MeanSquareError()
    print(mse_error(y_true, y_pred))
