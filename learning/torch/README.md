# Implementation using `torch`

# Modules
- [x] Softmax [[Docs](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#softmax)]
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$

# Loss Functions
- [x] Mean Square Error [[Docs](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)]
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}, i} - y_{\text{pred}, i})^2$$

- [x] Cross Entropy [[Docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)]
$$\text{Cross-Entropy Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{\text{true}, i,j} \cdot \log(y_{\text{pred}, i,j})$$

# TODO List
- [ ] Margin Loss
