# Implementation using `torch`

# Modules

- [x] Softmax [[Docs](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#softmax)]
  $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$
  - $x_i$ is the raw score or logit of the $i$ th class.
  - $N$ is the total number of classes.

# Loss Functions

- [x] Mean Square Error [[Docs](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)]
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}, i} - y_{\text{pred}, i})^2$$
  - $N$ is the number of samples or elements in the tensors.
  - $y_{\text{true}, i}$ is the true value of the $i$ th sample or element.
  - $y_{\text{pred}, i}$ is the predicted value of the $i$ th sample or element.

- [x] Cross
  Entropy [[Docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)]
  $$\text{Cross-Entropy Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{\text{true}, i,j} \cdot \log(y_
  {\text{pred}, i,j})$$
  - $N$ represents the batch size
  - $C$ represents the number of classes
  - $y_{\text{true}}$ represents the true probabilities
  - $y_{\text{pred}}$ represents the predicted probabilities

# TODO List

- [ ] Margin Loss
