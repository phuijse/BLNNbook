# Assignment: Improving VAE with Conditional Normalizing Flows

**Objective:** Implement, train and evaluate a flexible approximate posterior for VAE using conditional normalizing flows

**Requirements:**

- Use `torchvision.datasets.MNIST` to obtain the training and test data. Use digits 4 and 9 only (discard the others)
- Propose an architecture for this dataset and implement a VAE with a mean-field posterior. This will be the base model.
- Implement a more flexible posterior for VAE using conditional normalizing flows, consider compositions based on planar flows and neural spline flows (groups of two select only one flow).
- Compare the models in terms of ELBO, KL and reconstruction error as a function of the number of latent variables (consider at least 2 and 10), type of transformation and the length of the flow (consider at least 1, 2 and 5).
- Using a latent dimensionality of two, visualize the approximate posteriors and the generative space (see VAE lecture)
- Discuss your results!



**References**

1. https://arxiv.org/pdf/1505.05770.pdf
1. https://arxiv.org/pdf/1809.05861.pdf
1. https://docs.pyro.ai/en/stable/distributions.html#conditionaltransformeddistribution

**Deadline**

23:59, July 21th, 2021

```python

```
