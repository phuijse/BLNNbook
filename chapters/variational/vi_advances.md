# Recent advances on VI

For a survey of recent advances in Variational Inference, I highly recommend [Zhang et al 2018](https://arxiv.org/pdf/1711.05597.pdf). Some of the topics presented in this survey are discussed here

## Recap from previous lectures

We are interested in a posterior 

$$
p(\theta|\mathcal{D}) = \frac{p(\mathcal{D}|\theta) p(\theta)}{p(\mathcal{D})}
$$

which may be intractable. If that is the case we do approximate inference either through sampling (MCMC) or optimization (VI). 

In the latter we select a (simple) approximate posterior $q_\nu(\theta)$ and we optimize the parameters $\nu$ by maximizing the evidence lower bound (ELBO)

$$
\begin{split}
\log p(\mathcal{D}) \geq  \mathcal{L}(\nu) &= - \int q_\nu(\theta) \log \frac{q_\nu(\theta)}{p(\mathcal{D}|\theta) p (\theta)} d\theta  \\
&= \mathbb{E}_{\theta \sim q_\nu(\theta)} \left[\log p(\mathcal{D}|\theta) \frac{p(\theta)}{q_\nu(\theta)}\right ]   \\
&= \mathbb{E}_{\theta \sim q_\nu(\theta)} \left[\log p(\mathcal{D}|\theta)\right]- D_{KL}[q_\nu(\theta) || p(\theta)]   
\end{split}
$$

$$
\hat \nu = \text{arg}\max_\nu \mathbb{E}_{\theta \sim q_\nu(\theta)} \left[\log p(\mathcal{D}|\theta)\right]- D_{KL}[q_\nu(\theta) || p(\theta)] 
$$
which makes $q_\nu(\theta)$ close to $p(\theta|\mathcal{D})$

:::{note}

There is a trade-off between how flexible/expressive the posterior is and how simple is to approximate this expression

:::

In what follows we review different ways to improve VI

## More flexible approximate posteriors for VI


### Normalizing flows

One way to obtain a more flexible and still tractable posterior is to start with a simple distribution and apply a sequence of invertible transformations. This is the key idea behind normalizing flows. 

Let's say that $z\sim q(z)$ where $q$ is simple, *e.g.* standard gaussian, and that there is a smooth and invertible transformation $f$ such that $f^{-1}(f(z)) = z$

Then $z' = f(z)$ is a random variable too but its distribution is

$$
q_{z'}(z') = q(z) \left| \frac{\partial f^{-1}}{\partial z'} \right| = q(z) \left| \frac{\partial f}{\partial z} \right|^{-1}
$$

which is the original distribution times the inverse of jacobian of the transformation

And we can apply a chain of transformations $f_1, f_2, \ldots, f_K$ obtaining

$$
q_K(z_K) = q_0(z_0) \prod_{k=1}^K \left| \frac{\partial f_k}{\partial z_{k-1}} \right|^{-1}
$$

With this we can go from a simple Gaussian to more expressive/complex/multi-modal distributions 

Nowadays several types of flows exist in the literature, *e.g.* planar, radial, autoregresive. As example, see this work in which [normalizing flows were used to make the approximate posterior in VAE more expressive](https://arxiv.org/abs/1809.05861)

Key references:

- Dinh, Krueger and Bengio, ["NICE: Non-linear Independent Components Estimation"](https://arxiv.org/abs/1410.8516)
- Rezende and Mohamed, ["Variational Inference with Normalizing Flows"](https://arxiv.org/abs/1505.05770) 
- Kingma and Dhariwal, ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)


### Adding more structure

Another way to improve the variational approximation is to include auxiliary variables. For example in [Auxiliary Deep Generative Models](https://arxiv.org/abs/1602.05473) the VAE was extended by introducing a variable $a$ that does not modify the generative process but makes the approximate posterior more expressive

In this case the graphical model of the approximate posterior is $q(a, z |x) = q(z|a,x)q(a|x)$, so that the marginal $q(z|x)$ can fit more complicated posteriors. The graphical model of the generative process is $p(a,x,z) = p(a|x,z)p(x,z)$, *i.e.* under margnalization of $a$, $p(x,z)$ is recovered

The ELBO in this case is 

$$
\log p(x) = \int_z \int_a \log p(a, x, z) dz dz \geq \mathbb{E}_{\theta \sim q_\nu(a|z,x)} \left[\log \frac{p_\theta(a|x,z)p_\theta(x|z)p(z)}{q_\nu(a|x)q_\nu(z|a,x)}\right ]  \nonumber
$$



## Tigher bounds for the KL divergence

### Importance weighting

This is an idea based on importance sampling. Tigher bounds for the ELBO can be obtained by sampling several $z$ for a given $x$. This was explored for autoencoders in [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)

Let's say we sample independently $K$ times from the posterior, this yields progressively tighter lower bounds for the evidence:

$$
\mathcal{L}_k = \mathbb{E}_{z_K, \ldots, z_2, z_1 \sim q_\phi(z|x)} \log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}
$$

where $w_k = \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}$ are called the importance weights. Note that for $K=1$ we recover the VAE bound.

This tighter bound [has been shown to be equivalent to using the regular bound with a more complex posterior](https://arxiv.org/pdf/1808.09034.pdf). Recent discussion can be find in [Debiasing Evidence Approximations: On Importance-weighted Autoencoders and Jackknife Variational Inference](https://openreview.net/forum?id=HyZoi-WRb) and [Tighter Variational Bounds are Not Necessarily Better](https://arxiv.org/abs/1802.04537). 




## Other divergence measures



### $\alpha$ divergence 

The KL divergence is computationally-convenient but there are other options to measure how far two distributions are. For example the family of $\alpha$ divergences (Renyi's formulation)

$$
D_\alpha(p||q) = \frac{1}{\alpha -1} \log p(x)^\alpha q(x)^{1-\alpha} \,dx
$$

which is a generalization of the KL divergence: for $\alpha \to 1$ the KL is recovered

:::{note}

$\alpha$ represents a trade-of between the mass-covering and zero-forcing effects

:::

The $\alpha$ divergence has been explored for [VI recently](https://arxiv.org/pdf/1511.03243.pdf) and is [implemented in `numpyro`](https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.RenyiELBO)

### f divergence

The $\alpha$ divergence is a particular case of the f-divergence

$$
D_f(p||q) =  q(x) f \left ( \frac{p(x)}{q(x)} \right) \,dx
$$

where $f$ is a convex function with $f(0) = 1$. The KL is recovered for $f(z) = z \log(z)$

In general $f$ should defined such that the result in the bound does not depend on the marginal likelihood. [Wang, Liu and Liu, 2018](https://papers.nips.cc/paper/7816-variational-inference-with-tail-adaptive-f-divergence.pdf) proposed tail-adaptive f-divergence 



### Stein variational gradient descent (SVGD)

Another totally different approach is based on the **Stein operator**

$$
\mathcal{A}_p \phi(x) = \phi(x) \nabla_x \log p(x)  + \nabla_x \phi(x)
$$

where $p(x)$ is a distribution and $\phi(x) = [\phi_1(x), \phi_2(x), \ldots, \phi_d(x)]$ a smooth vector function

Under this following, known as the **Stein identity**, holds

$$
\mathbb{E}_{x\sim p(x)} \left [  \mathcal{A}_p \phi(x)  \right] = 0,
$$


Now, for another distribution $q(x)$ with the same support as $p$, we can write 

$$
\mathbb{E}_{x\sim q(x)} \left [ \mathcal{A}_p \phi(x) \right] - \mathbb{E}_{x\sim q(x)} \left [ \mathcal{A}_q \phi(x) \right]= \mathbb{E}_{x\sim q(x)} \left [ \phi(x) ( \nabla_x \log p(x) - \nabla_x \log q(x)) \right]
$$ 

from which the **Stein discrepancy** between two distributions is defined

$$
\sqrt{S(q, p)} = \max_{\phi\in \mathcal{F}} \mathbb{E}_{x\sim q(x)} \left [ \mathcal{A}_p \phi(x) \right]
$$

Which to actually work requires $\mathcal{F}$ to be broad enough

This is were kernels can be used. By taking an infinite amount of basis function $\phi(x)$ on the stein discrepancy it can be shown that the optimization is solved by

$$
\textbf{S}(q, p) = \mathbb{E}_{x, x' \sim q(x)} \left [ \mathcal{A}_p^x \mathcal{A}_p^{x'} \kappa(x, x')\right]
$$

where $\kappa$ is a kernel function, *e.g.* RBF or rational quadratic. From this one can use stochastic gradient descent. 

:::{seealso}

For a more in depth treatment see this [list of papers related to SVGD](https://www.cs.dartmouth.edu/~qliu/stein.html)

:::

### Operator VI

[Ranganath et al. 2016](https://arxiv.org/abs/1610.09033) proposed to replace the KL divergence as objective for VI with the Langevin-stein operator 

