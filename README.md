## Universidad Austral de Chile

# INFO3XX: Redes Neuronales y Aprendizaje Bayesiano

### Responsable: Pablo Huijse H, phuijse@inf.uach.cl

Un electivo del Magíster en Ingeniería Informática (MIN) de la UACh. Se recomienda mantener una copia local del material del curso clonando este repositorio. 

***
## Abstract

En este curso se estudiarán técnicas de programación probabilística que escalan a bases de datos masivas (Variational Inference), analizando los supuestos que las sustentan y las implementaciones existentes, con énfasis en su utilización para el entrenamiento de modelos de machine learning con interpretación Bayesiana. Se revisarán métodos probabilísticos de aprendizaje con aplicaciones en clasificación, regresión, extracción de variables latentes y clustering (agrupamiento). Se buscará acercar a los estudiantes al estado del arte del entrecruzamiento entre la teoría Bayesiana y el aprendizaje profundo (Bayesian Deep Learning) mediante clases expositivas, revisión de papers y tareas en lenguaje Python.


***
## Contenidos

- **Unidad 1:** Fundamentos
    - Distribuciones de probabilidad, Teorema de Bayes
    - Teoría de la Información (Entropía, Divergencia)
    - Modelo generativo con variable latente (GMM, PCA bayesiano)
    - Redes neuronales profundas (MLP, CNN, AE, GAN)
- **Unidad 2:** Inferencia variacional (VI) y redes neuronales bayesianas
    - Mean Field Variational Bayes y Aproximación de Laplace
    - Autoencoder variacional (VAE) (DP. Kingma, 2014), 
        - VAE condicional [(K. Sohn, 2016)](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) y semi-supervisado [(DP. Kingma, 2014)](https://arxiv.org/abs/1406.5298)
        - Avances recientes [(Y. Burda, 2016)](https://arxiv.org/abs/1509.00519) [(L. Maaloe, 2016)](https://arxiv.org/abs/1602.05473) [(T. Rainford, 2018)](https://arxiv.org/abs/1802.04537)
- **Unidad 3:** Procesos Gaussianos



***
## Bibliografía 


### Principal
1. D. Barber, "Bayesian reasoning and machine learning", Cambridge University Press, 2012, [**Libre!**](http://www.cs.ucl.ac.uk/staff/d.barber/brml/)
1. D. J. C. MacKay, "Information theory, inference and learning algorithms", Cambridge University Press, 2003, [**Libre!**](http://www.inference.org.uk/itprnn/book.html)
1. D. M. Blei, A. Kucukelbir, and J.D. McAuliffe. "Variational inference: A review for statisticians." Journal of the American Statistical Association 112(518), 859-877, 2017
1. D. Kingma, “Variational inference and deep learning: a new synthesis”, PhD thesis, University of Amsterdam, 2015

### Complementaria

1. K. P. Murphy, "Machine Learning: A probabilistic perspective", MIT Press, 2012
1. S. Theodoridis, "Machine Learning: A Bayesian and optimization perspective", Academic Press, 2015
1. A. Kendall and Y. Gal, “What uncertainties do we need in bayesian deep learning for computer vision?”, NIPS 2017, https://arxiv.org/pdf/1703.04977.pdf

***

## Software y librerías

- Lenguaje: [Python3](https://docs.python.org/3/)
- Ambiente: [IPython](https://ipython.org), [Jupyter](https://jupyter.org/)
- Librerías [PyTorch](https://pytorch.org/), [PyMC3](https://docs.pymc.io/)

#### (Recomendando) Ambiente de programación

Linux:
- Descargar e instalar *miniconda* de 64bits para Python 3.7: https://docs.conda.io/en/latest/miniconda.html
- Cargar variables de entorno (en este caso se instaló en home del usuario: 
    source ~/miniconda3/etc/profile.d/conda.sh
- Activar ambiente base
    conda activate
- Actualizar ambiente
    conda update --all
- Crear ambiente INFO3XX
    conda create -n INFO3XX
    conda activate INFO3XX
- Instalar Jupyter y PyTorch
    - Versión CPU
    
        conda install pip ipython jupyter mathjax pytorch-cpu==1.2.0
        
    -Versión GPU
        
        conda install pip ipython jupyter mathjax pytorch==1.2.0
- Instalar Pyro (solo a través de pip)
    
    pip install pyro-ppl




***
## Otros recursos


- [Programa del curso](https://docs.google.com/document/d/1EAEhxEz6LEDu7ux7NlD-ZLFRBq8fE-pxhkJf7W5y6JU/edit)

