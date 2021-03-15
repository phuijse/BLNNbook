## Universidad Austral de Chile

# INFO320: Neural Networks and Bayesian Learning

### Pablo Huijse H, phuijse at inf dot uach dot cl

An elective course at the Master o Informatics (MIN) of the Universidad Austral de Chile. Please mantain a local copy of this repository

***
## Abstract

In this course we will study probabilistic programming techniques that scale to massive datasets (Variational Inference), starting from the fundamentals and also reviewing existing implementations with emphasis on training deep neural network models that have a Bayesian interpretation. The objective is to present the student with the state of the art that lays at the intersection between the fields of Bayesian models and Deep Learning through lectures, paper reviews and practical exercises in Python (Pytorch plus Pyro)

<!-- #region -->
***
## Contents

- **Unit 1:** Fundamentals
    - [Probabilities and inference](lectures/01_fundamentals/01_probabilities_inference.ipynb)
    - [Information theory and generative models](lectures/01_fundamentals/02_information_theory.ipynb)
    - [Bayesian Linear Regression](lectures/01_fundamentals/03_linear_regression.ipynb)
    - [Function approximators and neural networks](lectures/01_fundamentals/04_neural_networks.ipynb)
- **Unit 2:** Bayesian Neural Networks
- **Unit 3:** Advanced/recent topics

***
## References 

### Mandatory:

1. Barber, D. (2012). [Bayesian reasoning and machine learning](http://www.cs.ucl.ac.uk/staff/d.barber/brml/). Cambridge University Press.
1. MacKay, D. J. (2003). [Information theory, inference and learning algorithms](http://www.inference.org.uk/mackay/itila/book.html). Cambridge university press.

### Suggested:

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). [Variational inference: A review for statisticians.](https://arxiv.org/abs/1601.00670) Journal of the American statistical Association, 112(518), 859-877.
1. Jospin, L. V., Buntine, W., Boussaid, F., Laga, H., & Bennamoun, M. (2020). [Hands-on Bayesian Neural Networks--a Tutorial for Deep Learning Users](https://arxiv.org/abs/2007.06823). arXiv preprint arXiv:2007.06823.
1. [Deep Bayes Moscow 2019](https://www.youtube.com/watch?v=SPgRVzfnESQ&list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW&index=2)


### Complementary

1. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.
1. Theodoridis, S. (2015). Machine learning: a Bayesian and optimization perspective. Academic press.


***
## Software and Libraries

- Programming language: [Python3](https://docs.python.org/3/)
- Development environment: [IPython](https://ipython.org) and [Jupyter](https://jupyter.org/)
- Libraries: [PyTorch](https://pytorch.org/) and [Pyro](http://pyro.ai/)
<!-- #endregion -->

### (Recommended) Installation using conda

- Install 64bits *miniconda* for Python 3.7: https://docs.conda.io/en/latest/miniconda.html
- Add conda to path (depends on where you installed it) 
    ```
    source ~/miniconda3/etc/profile.d/conda.sh
    ```
- Create environment. Do I have a nice and shiny NVIDIA GPU?
    - YES
        ```
        conda env create -f environment_gpu.yml
        ```
    - NO
        ```
        conda env create -f environment_cpu.yml
        ```
- Activate environment
    ```
    conda activate pyro-env
    ```
- Run jupyter from the environment
    ```
    jupyter notebook
    ```

