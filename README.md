# INFO320: Neural Networks and Bayesian Learning

An elective course at the [Master o Informatics (MIN) of the Universidad Austral de Chile](http://magister.inf.uach.cl/)

Professor: Pablo Huijse H, phuijse at inf dot uach dot cl

## Abstract

In this course we will study probabilistic programming techniques that scale to massive datasets (Variational Inference), starting from the fundamentals and also reviewing existing implementations with emphasis on training deep neural network models that have a Bayesian interpretation. The objective is to present the student with the state of the art that lays at the intersection between the fields of Bayesian models and Deep Learning through lectures, paper reviews and practical exercises in Python (Pytorch plus Pyro)

## Contents

- **Unit 1:** [Fundamentals](lectures/01_fundamentals)
- **Unit 2:** [Bayesian Neural Networks](lectures/02_bayesian_neural_networks)
- **Unit 3:** [Advanced/recent topics](lectures/03_advanced_topics)

## References 

### Mandatory:

- Barber, D. (2012). [Bayesian reasoning and machine learning](http://www.cs.ucl.ac.uk/staff/d.barber/brml/). Cambridge University Press.
- MacKay, D. J. (2003). [Information theory, inference and learning algorithms](http://www.inference.org.uk/mackay/itila/book.html). Cambridge university press.

### Suggested:

- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). [Variational inference: A review for statisticians.](https://arxiv.org/abs/1601.00670) Journal of the American statistical Association, 112(518), 859-877.
- Jospin, L. V., Buntine, W., Boussaid, F., Laga, H., & Bennamoun, M. (2020). [Hands-on Bayesian Neural Networks--a Tutorial for Deep Learning Users](https://arxiv.org/abs/2007.06823). arXiv preprint arXiv:2007.06823.
- [Deep Bayes Moscow 2019](https://www.youtube.com/watch?v=SPgRVzfnESQ&list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW&index=2)

### Complementary

- Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.
- Theodoridis, S. (2015). Machine learning: a Bayesian and optimization perspective. Academic press.

## Software and Libraries

- Programming language: [Python3](https://docs.python.org/3/)
- Development environment: [IPython](https://ipython.org) and [Jupyter](https://jupyter.org/)
- Libraries: [PyTorch](https://pytorch.org/) and [Pyro](http://pyro.ai/)


### Recommended installation 

To run these notebooks I recommend creating a conda environment, installing pytorch from the pytorch conda channel and installing `pyro` using pip

If you have a GPU you can copy and paste the following

```bash
conda create -n "info320"
conda install -c pyviz holoviews bokeh
conda install -c pytorch pytorch cudatoolkit ignite
conda install scipy scikit-learn selenium 
pip install pyro-ppl
```

