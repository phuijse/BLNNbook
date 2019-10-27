## Universidad Austral de Chile

# INFO3XX: Neural Networks and Bayesian Learning

### Pablo Huijse H, phuijse@inf.uach.cl

An elective course at the Master o Informatics (MIN) of the Universidad Austral de Chile. Please mantain a local copy of this repository

***
## Abstract

In this course we will study probabilistic programming techniques that scale to massive datasets (Variational Inference), starting from the fundamentales and also reviewing existing implementations with emphasis on training deep neural network models that have a Bayesian interpretation. The objective is to present the student with the state of the art that lays at the intersection between the fields of Bayesian models and Deep Learning through lectures, paper reviews and practical exercises in Python (Pytorch plus Pyro)


***
## Contents

- **Unit 1:** Fundamentals
    - [Probabilities and Inference](0_probabilities_inference.ipynb)
    - [Information Theory](2_function_approximators_neural_networks.ipynb)
    - [Function approximators and neural_networks](2_function_approximators_neural_networks.ipynb)
- **Unit 2:** Variational Inference (VI) and Bayesian Neural Networks
    - [Approximate Inference](3_approximate_inference.ipynb)
    - [Our first Bayesian Neural Net with Pyro](4_first_bayesian_nn_with_pyro.ipynb)
    - [Variational Autoencoder](5_variational_autoencoder.ipynb) 
        - VAE condicional [(K. Sohn, 2016)](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) y semi-supervisado [(DP. Kingma, 2014)](https://arxiv.org/abs/1406.5298)
        - Avances recientes [(Y. Burda, 2016)](https://arxiv.org/abs/1509.00519) [(L. Maaloe, 2016)](https://arxiv.org/abs/1602.05473) [(T. Rainford, 2018)](https://arxiv.org/abs/1802.04537)
- **Unit 3:** Gaussian Processes

***
## References 

1. D. Barber, "Bayesian reasoning and machine learning", Cambridge University Press, 2012, [**Libre!**](http://www.cs.ucl.ac.uk/staff/d.barber/brml/)
1. D. J. C. MacKay, "Information theory, inference and learning algorithms", Cambridge University Press, 2003, [**Libre!**](http://www.inference.org.uk/itprnn/book.html)
1. Y. Gal, ["Uncertainty in Deep Learning"](http://www.cs.ox.ac.uk/people/yarin.gal/website/), PhD thesis, University of Cambridge, 2015
1. D. P. Kingma, ["Variational inference and deep learning: a new synthesis"](http://dpkingma.com/), PhD thesis, University of Amsterdam, 2015
1. D. M. Blei, A. Kucukelbir, and J.D. McAuliffe. ["Variational inference: A review for statisticians"](https://arxiv.org/pdf/1601.00670.pdf) Journal of the American Statistical Association 112(518), 859-877, 2017

### Complementary

1. K. P. Murphy, "Machine Learning: A probabilistic perspective", MIT Press, 2012
1. S. Theodoridis, "Machine Learning: A Bayesian and optimization perspective", Academic Press, 2015
1. [INFO343: Redes Neuronales](https://docs.google.com/presentation/d/1IJ2n8X4w8pvzNLmpJB-ms6-GDHWthfsJTFuyUqHfXg8/edit?usp=sharing)

***

## Software and Libraries

- Programming language: [Python3](https://docs.python.org/3/)
- Development environment: [IPython](https://ipython.org) and [Jupyter](https://jupyter.org/)
- Libraries: [PyTorch](https://pytorch.org/) and [Pyro](http://pyro.ai/)

#### (Recommended) Installation using conda


- Install 64bits *miniconda* for Python 3.7: https://docs.conda.io/en/latest/miniconda.html
- Add conda to path (depends on where you installed it) 
    ```
    source ~/miniconda3/etc/profile.d/conda.sh
    ```
- Create an environment with the provided file
    ```
    conda env create -f environment.yml
    ```
- Activate environment
    ```
    conda activate pyro-env
    ```
- Run jupyter from the environment
    ```
    jupyter notebook
    ```

***
## Other resources


- [Programa del curso](https://docs.google.com/document/d/1EAEhxEz6LEDu7ux7NlD-ZLFRBq8fE-pxhkJf7W5y6JU/edit)

