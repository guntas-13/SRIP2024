# DDPM Implementation
## Reverse Diffusion Process over MNIST Digits

<div align = "center">
    <img src = "https://github.com/guntas-13/SRIP2024/blob/master/Media/x0_999.png" style="float: left;">
    <img src = "https://github.com/guntas-13/SRIP2024/blob/master/Media/x0_0.png">
</div>

<div align = "center">
<img src = "https://github.com/guntas-13/SRIP2024/blob/master/Media/out.gif">
</div>

# Denoising Diffusion Probabilistic Models
_(Posted Part I: 7 Jun 2024)_ <br>
_(Posted Part II: 8 Jun 2024)_ <br>
_Apologies for some formatting issues here, so check out blogs at [Blog Page](https://guntas-13.github.io/Blog/). Each equation here has to be tweaked in some way in order to render it._

As seen in the case of [Variational Autoencoders](https://guntas-13.github.io/Blog/posts/Generative/Maths.html), it all boils down to learning the probability distributions - $p(\textbf{z} | \textbf{x})$ the posterior abstraction of obtaining a hidden representation $\textbf{z}$ given some input image $\textbf{x}$ and the likelihood $p(\textbf{x} | \textbf{z})$ of generating the image samples given some hidden representation $\textbf{z}$.

Now the most crucial task in all these generative models is trying to understand to relate the objective that we are trying to acheive and what the model actually learns. We'll see the same confusing conclusion being established by the end of this blog and then we'll realise how beautifully all the mathematics and the tasks laid out make sense.


## Throwback to Variational Autoencoders

Like in the case of VAEs, we started off by approximating the actual $P(\textbf{z} | \textbf{x})$ through our probabilistic Encoder $Q_{\phi}(\textbf{z} | \textbf{x})$ and minimising the KL divergence between these two. But in order to establish the knowledge of the actual $P(\textbf{z} | \textbf{x})$, we went into maximising the log-likelihood of data samples $\textbf{x}$ and eventually made the encoder learn this distribution $Q_{\phi}(\textbf{z} | \textbf{x})$ to be as close to the standard normal $\mathcal{N}(\textbf{0}, \mathbb{I})$ as possible. Hence now drawing any $\textbf{z} \sim \mathcal{N}(\textbf{0}, \mathbb{I})$ we are sure of it being close to the $\textbf{z}$'s seen during training, allowing us to discard off the encoder entirely at inference. The way we setup the objective of making the actual and approximated distribution close to each other will stay same for Diffusion Models too and this would allow us to uncover more truth about the actual distribution itself.


![](https://github.com/guntas-13/SRIP2024/blob/master/Media/VAE1.png)
<p style="text-align: center; color: #5f9ea0;">Graphical Model of a Variation Autoencoder.</p>


## What are Diffusion Models?

For Diffusion Models instead of one latent variable $\textbf{z}$, we have $T$ latent variables of the form $\textbf{x}_1, \textbf{x}_2, \cdots , \textbf{x}_T$ of same dimension as the input image $\textbf{x}_0$, and the most interesting point is that the forward noising process is a deterministic Markov Chain, wherein Gaussian noise is added in gradual $T$ steps, defined as:

```math
\begin{equation}
\boxed{q(\textbf{x}_t | \textbf{x}_{t - 1}) = \mathcal{N}(\textbf{x}_{t}; \sqrt{1 - \beta_{t}} \textbf{x}_{t - 1}, \beta_{t} \mathbb{I})} 
\end{equation}
```

Here the variances are controlled by a scheduler $` \left \{ \beta_t \in (0, 1) \right \}_{t = 1}^T `$, which means for each noisy sample $` \textbf{x}_t `$ is sampled from a Gaussian with $` \mathbf{\mu}_q = \sqrt{1 - \beta_{t}} \textbf{x}_{t - 1} `$ and covariance matrix $` \mathbf{\Sigma}_q = \beta_{t} \mathbb{I} `$. The idea is then to learn the reverse denoising diffusion distribution $` q(\textbf{x}_{t - 1} | \textbf{x}_t) `$, which is also a **Markov Chain with learned Gaussian transitions** starting at $` p(\textbf{x}_T) \sim \mathcal{N}(\textbf{x}_T; \textbf{0}, \mathbb{I}) `$. Therefore it then becomes really important to understand the entire joint distribution $` p(\textbf{x}_1, \textbf{x}_2, \cdots, \textbf{x}_T) `$ denoted in shorthand as $` p(\textbf{x}_{0:T}) `$.

![](https://github.com/guntas-13/SRIP2024/blob/master/Media/Diff.png)
<p style="text-align: center; color: #5f9ea0;">Graphical Model of a Diffusion Process.</p>


## Prerequisites

### Joint & Conditional Distribution of $N$ RVs and Bayes' Rule

A joint distribution over $N$ random variables assigns probabilities to all the events involving these $N$ random variables^[$k^N$ values if each RV can take $k$ values], denoted as

$$ P(X_1, X_2, X_3, \cdots, X_N) $$

Now starting off with just two RVs, the conditional probabilities $P(X_1 | X_2)$ and $P(X_2 | X_1)$ can be calculated from the joint distribution as:

$$
\begin{align}
P(X_1 | X_2) = \frac{P(X_1, X_2)}{P(X_2)} && P(X_2 | X_1) = \frac{P(X_1, X_2)}{P(X_1)} 
\end{align}
$$

Conveniently this allows us to write $P(X_1, X_2) = P(X_2 | X_1) P(X_1)$. And similarly for $N$ random variables

$$
\begin{align}
P(X_1, X_2, X_3, \cdots, X_N) &= P(X_2, X_3, \cdots, X_N | X_1) \cdot P(X_1) \\
&= P(X_3, \cdots, X_N | X_1, X_2) \cdot P(X_2 | X_1) \cdot P(X_1) \\
&= P(X_4, \cdots, X_N | X_1, X_2, X_3) \cdot P(X_3 | X_2, X_1) \cdot P(X_2 | X_1) \cdot P(X_1)
\end{align}
$$

Expanding by chain rule we get

$$ \boxed{P(X_1, X_2, X_3, \cdots, X_N) = P(X_1) \cdot \prod_{i = 2}^{N} P(X_i | X_1^{i - 1})} $$

where $X_1^{i - 1} = X_1, X_2, X_3, \cdots, X_{i - 1}$ and then using the joint-conditional-marginal formula, we get the **Bayes' Rule** as

$$ P(X_2 | X_1) = \frac{P(X_1 | X_2) \cdot P(X_2)}{P(X_1)} $$


### Markov-Diffusion Process and Reparametrization

The special property of a Markov Process is that the future state is dependent **only on previous state**, which means

```math
\begin{equation}
P(\textbf{x}_t | \textbf{x}_{t - 1}, \cdots, \textbf{x}_0) = P(\textbf{x}_t | \textbf{x}_{t - 1})
\end{equation}
```

the utility of this is that using the chain rule of joint probability, we may simply write for all our diffusion process forward steps $`q(\textbf{x}_{1 : T} | \textbf{x}_0)`$ as

```math
\begin{align}
q(\textbf{x}_{1 : T} | \textbf{x}_0) = \prod_{t = 1}^{T} q(\textbf{x}_t | \textbf{x}_{t - 1}) && q(\textbf{x}_t | \textbf{x}_{t - 1}) = \mathcal{N}(\textbf{x}_{t}; \sqrt{1 - \beta_{t}} \textbf{x}_{t - 1}, \beta_{t} \mathbb{I})
\end{align}
```

**Diffusion** is the process of converting samples from a **complex distribution** (the data here) $`\textbf{x}_0 \sim q(\textbf{x}_0)`$ to samples of a **simple distribution** (isotropic Gaussian noise) $`\textbf{x}_T \sim \mathcal{N}(\textbf{0}, \mathbb{I})`$. One can also observe that there is a $`\color{purple}{\text{deterministic}}`$ and a $`\color{blue}{\text{stochastic}}`$ component even in our case. Since any RV can be reparametrized as $`Z = \sigma X + \mu`$, hence we denote the $`\textbf{x}_t`$ being drawn from $`q(\textbf{x}_t | \textbf{x}_{t - 1})`$ as

```math
\begin{equation}
\boxed{\textbf{x}_t = \color{purple}{\sqrt{1 - \beta_t} \textbf{x}_{t - 1}} + \color{blue}{\sqrt{\beta_t} \epsilon_{t - 1}}}
\end{equation}
```

where $`\epsilon_{t - 1}, \epsilon_{t - 2}, \cdots \sim \mathcal{N}(\textbf{0}, \mathbb{I})`$

![](https://github.com/guntas-13/SRIP2024/blob/master/Media/Diff1.png)
<p style="text-align: center; color: #5f9ea0;">Forward and Reverse Diffusion Processes.</p>


## Understanding the Forward Markov Process

One might wonder why does following the above said markov chain of gaussians lead to $\textbf{x}_T \sim \mathcal{N}(\textbf{0}, \mathbb{I})$. To understand this let's take arbitrary constants for the above

```math
\begin{equation}
\textbf{x}_t = \sqrt{\alpha} \textbf{x}_{t - 1} + \sqrt{\beta} \epsilon_{t - 1}
\end{equation}
```

Since we know that $`\textbf{x}_T \sim \mathcal{N}(\textbf{0}, \mathbb{I})`$ so let's open up the formula from this end

```math
\begin{align}
\textbf{x}_T &= \sqrt{\alpha} \textbf{x}_{T - 1} + \sqrt{\beta} \mathcal{N}(\textbf{0}, \mathbb{I}) \\
&= \sqrt{\alpha} (\sqrt{\alpha} \textbf{x}_{T - 2} + \sqrt{\beta} \mathcal{N}(\textbf{0}, \mathbb{I})) + \sqrt{\beta} \mathcal{N}(\textbf{0}, \mathbb{I}) \\
&= (\sqrt{\alpha})^2 \textbf{x}_{T - 2} + \sqrt{\alpha} \sqrt{\beta} \mathcal{N}(\textbf{0}, \mathbb{I}) + \sqrt{\beta} \mathcal{N}(\textbf{0}, \mathbb{I}) \\
\cdots \\
&= (\sqrt{\alpha})^T \textbf{x}_0 + \sqrt{\beta} ((\sqrt{\alpha})^{T - 1} \mathcal{N}(\textbf{0}, \mathbb{I}) + (\sqrt{\alpha})^{T - 2} \mathcal{N}(\textbf{0}, \mathbb{I}) + \cdots + \mathcal{N}(\textbf{0}, \mathbb{I}))
\end{align}
```

We can combine the independent Gaussians into one Gaussian (Two Gaussians with different variances, $`\mathcal{N}(\textbf{0}, \sigma_1^2\mathbb{I})`$ and $`\mathcal{N}(\textbf{0}, \sigma_2^2\mathbb{I})`$ can be merged to a new Gaussian distribution $`\mathcal{N}(\textbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbb{I})`$) as they have **variances** as $`(\beta \alpha^{T - 1}, \beta \alpha^{T - 2}, \cdots, \beta \alpha, \beta)`$ with $`\sigma^2 = \beta \frac{1 - \alpha^T}{1 - \alpha}`$. Notice that as $`T \to \infty, (\sqrt{\alpha})^T \to 0`$ and $`\textbf{x}_T \to \mathcal{N}(\textbf{0}, \mathbb{I})`$ only when $`\alpha = 1 - \beta`$.


### Do we traverse for all $T$ steps?
Certainly Not! Here's how the Markov Process allows us to reach any $`\textbf{x}_t`$ from the image $`\textbf{x}_0`$. Let $`\alpha_t = 1 - \beta_t`$

```math
\begin{align}
\textbf{x}_t &= \sqrt{\alpha_t} \textbf{x}_{t - 1} + \sqrt{1 - \alpha_t} \mathcal{N}(\textbf{0}, \mathbb{I}) \\
&= \sqrt{\alpha_t} (\sqrt{\alpha_{t - 1}} \textbf{x}_{t - 2} + \sqrt{1 - \alpha_{t - 1}} \mathcal{N}(\textbf{0}, \mathbb{I})) + \sqrt{1 - \alpha_t} \mathcal{N}(\textbf{0}, \mathbb{I}) \\
&= (\sqrt{\alpha_t \alpha_{t - 1}}) \textbf{x}_{t - 2} + (\sqrt{\alpha_t} \sqrt{1 - \alpha_{t - 1}}) \mathcal{N}(\textbf{0}, \mathbb{I}) + (\sqrt{1 - \alpha_t}) \mathcal{N}(\textbf{0}, \mathbb{I})
\end{align}
```

Combine the independent Gaussians with $`\sigma^2 = \alpha_t (1 - \alpha_{t - 1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t - 1}`$, hence

```math
\begin{align}
\textbf{x}_t &= (\sqrt{\alpha_t \alpha_{t - 1}}) \textbf{x}_{t - 2} + (\sqrt{1 - \alpha_t \alpha_{t - 1}})\mathcal{N}(\textbf{0}, \mathbb{I}) \\
\cdots \\
&= (\sqrt{\alpha_t \alpha_{t - 1} \cdots \alpha_2 \alpha_1}) \textbf{x}_0 + (\sqrt{1 - \alpha_t \alpha_{t - 1} \cdots \alpha_2 \alpha_1}) \mathcal{N}(\textbf{0}, \mathbb{I})
\end{align}
```

With $`\bar{\alpha_t} = \prod^t \alpha_i`$, we finally get

```math
\begin{equation}
\boxed{\textbf{x}_t = (\sqrt{\bar{\alpha_t}}) \textbf{x}_0 + (\sqrt{1 - \bar{\alpha_t}}) \mathcal{N}(\textbf{0}, \mathbb{I})}
\end{equation}
```

```math
\begin{equation}
\boxed{q(\textbf{x}_t | \textbf{x}_0) = \mathcal{N}(\textbf{x}_t; (\sqrt{\bar{\alpha_t}}) \textbf{x}_0, (1 - \bar{\alpha_t}) \mathbb{I})}
\end{equation}
```

## The Crucial Reverse Diffusion Process

If we can reverse the above process and sample from $`q(\textbf{x}_{t - 1} | \textbf{x}_t)`$, we will be able to recreate the true sample from a Gaussian noise input, $`\textbf{x}_T \sim \mathcal{N}(\textbf{0}, \mathbb{I})`$ . Note that if $`\beta_t`$ is small enough, $`q(\textbf{x}_{t - 1} | \textbf{x}_t)`$ will also be Gaussian. Unfortunately, we cannot easily estimate $`q(\textbf{x}_{t - 1} | \textbf{x}_t)`$ because it needs to use the entire dataset and therefore we need to learn a model $`p_{\theta}`$ to approximate these conditional probabilities in order to run the reverse diffusion process. The actual reverse distribution

```math
\begin{equation}
q(\textbf{x}_{t - 1} | \textbf{x}_t) = \mathcal{N}(\textbf{x}_{t - 1}; \boldsymbol{\mu}_q, \mathbf{\Sigma}_q)
\end{equation}
```

And the approximated distribution can be represented as

```math
\begin{align}
p_{\theta}(\textbf{x}_{0 : T}) = p(\textbf{x}_T) \prod_{t = 1}^T p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t) && p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t) = \mathcal{N}(\textbf{x}_{t - 1}; \boldsymbol{\mu}_{\theta}(\textbf{x}_t, t), \mathbf{\Sigma}_{\theta}(\textbf{x}_t, t))
\end{align}
```

Before moving onto defining the objective to find the approximate $`p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)`$, its noteworthy to understand the actual reverse process distribution $`q(\textbf{x}_{t - 1} | \textbf{x}_t)`$. As stated by the DDPM paper, the reverse conditional distribution is tractable when condition on $`\textbf{x}_0`$ and since this a Markov Process, we can safely introduce this $`\textbf{x}_0`$ in the joint conditional part and then we may expand it by **Bayes' Rule**

```math
\begin{equation}
q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) = \frac{q(\textbf{x}_t | \textbf{x}_{t - 1}, \textbf{x}_0) \cdot q(\textbf{x}_{t - 1} | \textbf{x}_0)}{q(\textbf{x}_t | \textbf{x}_0)}
\end{equation}
```

Notice that all of these are forward processes, and using $`\mathcal{N}(\textbf{x}; \boldsymbol{\mu}, \boldsymbol{\sigma}^2) \propto \text{exp}(-\frac{1}{2} \frac{(\textbf{x} - \boldsymbol{\mu})^2}{\boldsymbol{\sigma}^2})`$

```math
\begin{equation}
q(\textbf{x}_t | \textbf{x}_{t - 1}, \textbf{x}_0) = q(\textbf{x}_t | \textbf{x}_{t - 1}) = \mathcal{N}(\textbf{x}_t; \sqrt{\alpha_t}\textbf{x}_{t - 1}, (1 - \alpha_t) \mathbb{I}) \propto \text{exp} \left( -\frac{1}{2} \frac{(\textbf{x}_t - \sqrt{\alpha_t}\textbf{x}_{t - 1})^2}{(1 - \alpha_t)} \right)
\end{equation}
```

```math
\begin{equation}
q(\textbf{x}_{t - 1} | \textbf{x}_0) = \mathcal{N}(\textbf{x}_{t - 1}; \sqrt{\bar{\alpha}_{t - 1}}\textbf{x}_0, (1 - \bar{\alpha}_{t - 1}) \mathbb{I}) \propto\text{exp} \left( -\frac{1}{2} \frac{(\textbf{x}_{t - 1} - \sqrt{\bar{\alpha}_{t - 1}}\textbf{x}_0)^2}{(1 - \bar{\alpha}_{t - 1})} \right)
\end{equation}
```

```math
\begin{equation}
q(\textbf{x}_t | \textbf{x}_0) = \mathcal{N}(\textbf{x}_t; \sqrt{\bar{\alpha}_t}\textbf{x}_0, (1 - \bar{\alpha}_t) \mathbb{I}) \propto\text{exp} \left( -\frac{1}{2} \frac{(\textbf{x}_t - \sqrt{\bar{\alpha}_t}\textbf{x}_0)^2}{(1 - \bar{\alpha}_t)} \right)
\end{equation}
```

Hence, we may combine all these to get a single Gaussian for $`q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0)`$ as

```math
\begin{align}
q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) &\propto \textbf{exp} \left(-\frac{1}{2} \left( \frac{(\textbf{x}_t - \sqrt{\alpha_t}\textbf{x}_{t - 1})^2}{(1 - \alpha_t)} + \frac{(\textbf{x}_{t - 1} - \sqrt{\bar{\alpha}_{t - 1}}\textbf{x}_0)^2}{(1 - \bar{\alpha}_{t - 1})} + \frac{(\textbf{x}_t - \sqrt{\bar{\alpha}_t}\textbf{x}_0)^2}{(1 - \bar{\alpha}_t)} \right) \right) \\
&= \textbf{exp} \left(-\frac{1}{2} \left( \frac{\textbf{x}_t^2 - 2 \sqrt{\alpha_t} \textbf{x}_t \color{blue}{\textbf{x}_{t - 1}} \color{black}{+ \alpha_t} \color{green}{\textbf{x}^2_{t - 1}}}{(1 - \alpha_t)} + \frac{\bar{\alpha}_{t - 1}\textbf{x}_0^2 - \sqrt{\bar{\alpha}_{t - 1}}\textbf{x}_0 \color{blue}{\textbf{x}_{t - 1}} \color{black}{+} \color{green}{\textbf{x}_{t - 1}^2}}{(1 - \bar{\alpha}_{t - 1})} + \frac{(\textbf{x}_t - \sqrt{\bar{\alpha}_t}\textbf{x}_0)^2}{(1 - \bar{\alpha}_t)} \right) \right) \\
\end{align}
```

```math
\begin{align}
&= \textbf{exp} \left(-\frac{1}{2} \left( \left( \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t - 1}} \right) \color{green}{\textbf{x}^2_{t - 1}} \color{black}{-2 \left( \frac{\sqrt{\alpha_t} \textbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t - 1}} \textbf{x}_0}{1 - \bar{\alpha}_{t - 1}} \right)} \color{blue}{\textbf{x}_{t - 1}} \color{black}{+ F(\textbf{x}_t, \textbf{x}_0)} \right) \right) \\
&\propto \textbf{exp} \left(-\frac{1}{2 \boldsymbol{\sigma}^2} \left( \color{green}{\textbf{x}^2_{t - 1}} \color{black}{-2 \boldsymbol{\mu}} \color{blue}{\textbf{x}_{t - 1}} \color{black}{+ \boldsymbol{\mu}^2} \right) \right) \\
&\propto \textbf{exp} \left( -\frac{1}{2 \boldsymbol{\sigma}^2} \left( \textbf{x}_{t - 1} - \boldsymbol{\mu} \right)^2 \right)
\end{align}
```

where the variance can be written as

```math
\begin{align}
\boldsymbol{\sigma}^2 &= 1 / \left( \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t - 1}} \right) \\
&= \frac{(1 - \alpha_t) \cdot (1 - \bar{\alpha}_{t - 1})}{(1 - \bar{\alpha}_t)}
\end{align}
```

and the mean as the weighted mean of $`\textbf{x}_t`$ and $`\textbf{x}_0`$ be written as

```math
\begin{align}
\boldsymbol{\mu} &= \frac{(1 - \alpha_t) \cdot (1 - \bar{\alpha}_{t - 1})}{(1 - \bar{\alpha}_t)} \left( \frac{\sqrt{\alpha_t}}{1 - \alpha_t} \textbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_{t - 1}}\textbf{x}_0 \right) \\
&= \frac{(1 - \bar{\alpha}_{t - 1}) \sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \textbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_t}\textbf{x}_0 \\
&= \frac{(1 - \bar{\alpha}_{t - 1}) \sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \textbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_t} \left( \frac{1}{\sqrt{\bar{\alpha}_t}} (\textbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_t) \right) \\
& (\because \textbf{x}_t = \sqrt{\bar{\alpha_t}} \textbf{x}_0 + (\sqrt{1 - \bar{\alpha_t}}) \boldsymbol{\epsilon}_t ) \\
&= \frac{1}{\sqrt{\alpha_t}} \left(\textbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right)
\end{align}
```

Finally we have the original reverse distribution $`q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0)`$ as 

```math
\begin{equation}
\color{MidnightBlue}{\boxed{q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) = q(\textbf{x}_{t - 1} | \textbf{x}_t) = \mathcal{N}(\textbf{x}_{t - 1}; \boldsymbol{\mu}_q(\textbf{x}_0, \textbf{x}_t), \mathbf{\Sigma}_q(t))}}
\end{equation}
```

```math
\begin{equation}
\color{RubineRed}{\boxed{\mathbf{\Sigma}_q(t) = \frac{(1 - \alpha_t) \cdot (1 - \bar{\alpha}_{t - 1})}{(1 - \bar{\alpha}_t)}}}
\end{equation}
```

```math
\begin{equation}
\color{teal}{\boxed{\boldsymbol{\mu}_q(\textbf{x}_t, \textbf{x}_0) = \frac{(1 - \bar{\alpha}_{t - 1}) \sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \textbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_t}\textbf{x}_0}}
\end{equation}
```

```math
\begin{equation}
\color{teal}{\boxed{\boldsymbol{\mu}_q(t) = \frac{1}{\sqrt{\alpha_t}} \left(\textbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right)}}
\end{equation}
```


## The Loss Function

### Defining the Loss Function
As discussed before we will follow the same methodology as done in VAEs of learning the approximate reverse distribution $`p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)`$ by maximizing the expected log-likelihood of the observed data $`p_{\theta}(\textbf{x}_0)`$ for $`\textbf{x}_0 \sim q(\textbf{x}_0)`$

![](https://github.com/guntas-13/SRIP2024/blob/master/Media/Jensen.png)
<p style="text-align: center; color: #5f9ea0;">Jenson's Inequality.</p>

Using Jensen's Inequality over the $`\log`$ function (convex function), hence the expectation of $`\log`$ is lesser than equal to the $`\log`$ of expectation.


```math
\begin{align}
L &= \mathbb{E}_{\textbf{x}_0 \sim q(\textbf{x}_0)} \left[ \log p_{\theta}(\textbf{x}_0) \right] \\
&= \mathbb{E}_{q(\textbf{x}_0)} \left[ \log \int p_{\theta}(\textbf{x}_{0 : T}) d\textbf{x}_{1 : T} \right] \\
&= \mathbb{E}_{q(\textbf{x}_0)} \left[ \log \int q(\textbf{x}_{1 : T} | \textbf{x}_0) \times \frac{p_{\theta}(\textbf{x}_{0 : T})}{q(\textbf{x}_{1 : T} | \textbf{x}_0)} d\textbf{x}_{1 : T} \right] \\
&= \mathbb{E}_{q(\textbf{x}_0)} \left[ \log \left ( \mathbb{E}_{q(\textbf{x}_{1 : T} | \textbf{x}_0)} \left[ \frac{p_{\theta}(\textbf{x}_{0 : T})}{q(\textbf{x}_{1 : T} | \textbf{x}_0)} \right) \right] \right] \\
&\ge \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[\color{OrangeRed}{ \log \frac{p_{\theta}(\textbf{x}_{0 : T})}{q(\textbf{x}_{1 : T} | \textbf{x}_0)}} \color{black}{} \right] \\
\end{align}
```

Notice that both these terms are joint probability distributions with $`\color{OrangeRed}{q(\textbf{x}_{1 : T} | \textbf{x}_0)}`$ being the actual forward process and $`\color{OrangeRed}{p_{\theta}(\textbf{x}_{0 : T})}`$ the approximate reverse process. Expanding these terms out

```math
\begin{equation}
p_{\theta}(\textbf{x}_{0 : T}) = p_{\theta}(\textbf{x}_T) \cdot \prod_{t = 1}^T p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)
\end{equation}
```


Further we'll condition the forward process on $`\textbf{x}_0`$ as it would later allow us to expand terms using **Bayes' Rule** $`q(\textbf{x}_t | \textbf{x}_{t - 1}, \textbf{x}_0) = \frac{q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) \cdot q(\textbf{x}_t | \textbf{x}_0)}{q(\textbf{x}_{t - 1} | \textbf{x}_0)}`$


```math
\begin{align}
q(\textbf{x}_{1 : T} | \textbf{x}_0) &= \prod_{t = 1}^T q(\textbf{x}_t | \textbf{x}_{t - 1}) \\
&= q(\textbf{x}_1 | \textbf{x}_0) \cdot \prod_{t = 2}^T q(\textbf{x}_t | \textbf{x}_{t - 1}, \color{orange}{\textbf{x}_0} \color{black}{}) \\
&= q(\textbf{x}_1 | \textbf{x}_0) \cdot \prod_{t = 2}^T \frac{q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) \cdot q(\textbf{x}_t | \textbf{x}_0)}{q(\textbf{x}_{t - 1} | \textbf{x}_0)}
\end{align}
```

```math
\begin{align}
&= q(\textbf{x}_1 | \textbf{x}_0) \cdot \frac{q(\textbf{x}_{T - 1} | \textbf{x}_T, \textbf{x}_0)q(\textbf{x}_T | \textbf{x}_0) \cdot q(\textbf{x}_{T - 2} | \textbf{x}_{T - 1}, \textbf{x}_0)q(\textbf{x}_{T - 1} | \textbf{x}_0) \cdots q(\textbf{x}_2 | \textbf{x}_3, \textbf{x}_0)q(\textbf{x}_3 | \textbf{x}_0) \cdot q(\textbf{x}_1 | \textbf{x}_2, \textbf{x}_0)q(\textbf{x}_2 | \textbf{x}_0)}{q(\textbf{x}_{T - 1} | \textbf{x}_0) \cdot q(\textbf{x}_{T - 2} | \textbf{x}_0) \cdots  q(\textbf{x}_2 | \textbf{x}_0) \cdot q(\textbf{x}_1 | \textbf{x}_0)} \\
&= \cancel{q(\textbf{x}_1 | \textbf{x}_0)} \cdot \frac{q(\textbf{x}_{T - 1} | \textbf{x}_T, \textbf{x}_0)q(\textbf{x}_T | \textbf{x}_0) \cdot q(\textbf{x}_{T - 2} | \textbf{x}_{T - 1}, \textbf{x}_0)\cancel{q(\textbf{x}_{T - 1} | \textbf{x}_0)} \cdots q(\textbf{x}_2 | \textbf{x}_3, \textbf{x}_0)\cancel{q(\textbf{x}_3 | \textbf{x}_0)} \cdot q(\textbf{x}_1 | \textbf{x}_2, \textbf{x}_0)\cancel{q(\textbf{x}_2 | \textbf{x}_0)}}{\cancel{q(\textbf{x}_{T - 1} | \textbf{x}_0)} \cdot \cancel{q(\textbf{x}_{T - 2} | \textbf{x}_0)} \cdots  \cancel{q(\textbf{x}_2 | \textbf{x}_0)} \cdot \cancel{q(\textbf{x}_1 | \textbf{x}_0)}} \\
\Aboxed{q(\textbf{x}_{1 : T} | \textbf{x}_0) &= q(\textbf{x}_T | \textbf{x}_0) \cdot \prod_{t = 2}^T q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0)}
\end{align}
```

Substituting it all for the new lower-bound loss

```math
\begin{align}
L_{N} &= \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[\log \frac{p_{\theta}(\textbf{x}_{0 : T})}{q(\textbf{x}_{1 : T} | \textbf{x}_0)} \right] \\
&= \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[\log \frac{p_{\theta}(\textbf{x}_T) \cdot \prod_{t = 1}^T p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)}{q(\textbf{x}_1 | \textbf{x}_0) \cdot \prod_{t = 2}^T q(\textbf{x}_t | \textbf{x}_{t - 1}, \textbf{x}_0)} \right] \\
&= \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[\log \frac{p_{\theta}(\textbf{x}_T) \cdot \prod_{t = 1}^T p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)}{q(\textbf{x}_1 | \textbf{x}_0) \cdot \prod_{t = 2}^T \frac{q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) \cdot q(\textbf{x}_t | \textbf{x}_0)}{q(\textbf{x}_{t - 1} | \textbf{x}_0)} } \right] \\
\end{align}
```

```math
\begin{align}
&= \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[\log \frac{p_{\theta}(\textbf{x}_T) \cdot \prod_{t = 1}^T p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)}{q(\textbf{x}_T | \textbf{x}_0) \cdot \prod_{t = 2}^T q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) } \right] \\
&= \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[\log \frac{p_{\theta}(\textbf{x}_T)}{q(\textbf{x}_T | \textbf{x}_0)} + \log p_{\theta}(\textbf{x}_0 | \textbf{x}_1) + \sum_{t = 2}^T \log \frac{p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)}{q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0)} \right] \\
&= \log p_{\theta}(\textbf{x}_0 | \textbf{x}_1) + \mathbb{E}_{q(\textbf{x}_{0 : T})} \left[ \sum_{t = 2}^T \log \frac{p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)}{q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0)} \right] \\
\Aboxed{L_N &= \sum_{t = 2}^T D_{KL} \left(p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t) \parallel q(\textbf{x}_{t - 1} | \textbf{x}_t, \textbf{x}_0) \right) + \log p_{\theta}(\textbf{x}_0 | \textbf{x}_1)}
\end{align}
```

### Reparametrization of the Loss Function

The above loss function aims to bring the actual reverse $`q(\textbf{x}_{t - 1} | \textbf{x}_t)`$ and approximated reverse distributions $`p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t)`$ as close as possible by the means of the $`T - 1`$ $`KL`$ Divergence terms.  Since we approximate the reverse process using a neural network, the Divergence terms would imply that we want their means to be as close as possible. For two Gaussians with same covariance matrices $`p = \mathcal{N}(\textbf{x}; \boldsymbol{\mu}_1, \mathbf{\Sigma})`$ and $`q = \mathcal{N}(\textbf{x}; \boldsymbol{\mu}_2, \mathbf{\Sigma})`$, their $`D_{KL}(p \parallel q) = \frac{1}{2 \lVert \mathbf{\Sigma} \rVert^2_2} \lVert \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2 \rVert^2`$

```math
\begin{equation}
p_{\theta}(\textbf{x}_{t - 1} | \textbf{x}_t) = \mathcal{N}(\textbf{x}_{t - 1}; \boldsymbol{\mu}_{\theta}(\textbf{x}_t, t), \mathbf{\Sigma}_{\theta}(\textbf{x}_t, t))
\end{equation}
```

```math
\begin{equation}
q(\textbf{x}_{t - 1} | \textbf{x}_t) = \mathcal{N}(\textbf{x}_{t - 1}; \left( \frac{(1 - \bar{\alpha}_{t - 1}) \sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \textbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_t}\textbf{x}_0 \right), \frac{(1 - \alpha_t) \cdot (1 - \bar{\alpha}_{t - 1})}{(1 - \bar{\alpha}_t)} \mathbb{I})
\end{equation}
```

As stated in the paper, take both their covariances to be the same, hence each term of the loss function $`L_t`$ can be written as

```math
\begin{equation}
\color{BlueViolet}{\boxed{L_t = \mathbb{E}_{\textbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{1}{2 \lVert \mathbf{\Sigma}_{\theta}(\textbf{x}_t, t) \rVert^2_2} \lVert \boldsymbol{\mu}_{\theta}(\textbf{x}_t, t) - \boldsymbol{\mu}_t(\textbf{x}_t, \textbf{x}_0) \rVert^2 \right] }}
\end{equation}
```


The authors, however, define this in terms of the noise prediction. Since $`\boldsymbol{\mu}_q(\textbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(\textbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right)`$, hence for the approximate reverse process distribution, we may write

```math
\begin{equation}
\boldsymbol{\mu}_{\theta}(\textbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(\textbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta}(\textbf{x}_t, t) \right)
\end{equation}
```

Hence the objective can be re-written as

```math
\begin{align}
L_t &= \mathbb{E}_{\textbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{1}{2 \lVert \mathbf{\Sigma}_{\theta} \rVert^2_2} \lVert \frac{1}{\sqrt{\alpha_t}} \left(\textbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta}(\textbf{x}_t, t) \right) - \frac{1}{\sqrt{\alpha_t}} \left(\textbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right) \rVert^2 \right] \\
&= \mathbb{E}_{\textbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha}_t) \lVert \mathbf{\Sigma}_{\theta} \rVert^2_2} \lVert \boldsymbol{\epsilon}_{\theta}(\textbf{x}_t, t) - \boldsymbol{\epsilon}_t \rVert^2 \right]
\end{align}
```

```math
\begin{equation}
\color{PineGreen}{\boxed{L_t = \mathbb{E}_{\textbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha}_t) \lVert \mathbf{\Sigma}_{\theta} \rVert^2_2} \lVert \boldsymbol{\epsilon}_{\theta}((\sqrt{\bar{\alpha_t}}) \textbf{x}_0 + (\sqrt{1 - \bar{\alpha_t}})\boldsymbol{\epsilon}_t , t) - \boldsymbol{\epsilon}_t \rVert^2 \right]}}
\end{equation}
```


Notice how beautifully it wraps down to just making the model to learn to approximate the noising $`\boldsymbol{\epsilon}_{\theta}(\textbf{x}_t, t)`$ process over any $`\textbf{x}_t`$ to the actual noise $`\boldsymbol{\epsilon}_t \sim \mathcal{N}(\textbf{0}, \mathbb{I})`$. Hence this quite so weird learning process lets us learn the denoising reverse distribution.

### Modified Objective
The authors further found that training works better by dropping off the constant term entirely, so the final objective is

```math
\begin{equation}
\color{OrangeRed}{\boxed{L_t^{\text{Simple}} = \mathbb{E}_{t \sim [1, T], \textbf{x}_0, \boldsymbol{\epsilon}_t} \left[\lVert \boldsymbol{\epsilon}_{\theta}((\sqrt{\bar{\alpha_t}}) \textbf{x}_0 + (\sqrt{1 - \bar{\alpha_t}})\boldsymbol{\epsilon}_t , t) - \boldsymbol{\epsilon}_t \rVert^2 \right]}}
\end{equation}
```

![](https://github.com/guntas-13/SRIP2024/blob/master/Media/DiffAlgo.png)
<p style="text-align: center; color: #5f9ea0;">Training and Inference Algorithms.</p>

Great [YouTube tutorial](https://www.youtube.com/watch?v=H45lF4sUgiE) by **ExplainingAI** helped understand the whole concept!



<h1 align = "center"> DCGAN MNIST </h1>

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/DCGAN_MNIST.gif" style="border:0;">
</p>

<h1 align = "center"> DCGAN CelebA </h1>

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/DCGAN_CelebA.gif" style="border:0;">
</p>

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/DCGAN_Celeb.png" style="border:0;">
</p>

<h1 align = "center"> Interpolation in the Latent Space </h1>

```math
\begin{equation}
\textbf{w}^* = \lambda \textbf{a} + (1 - \lambda) \textbf{b}
\end{equation}
```

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/Inter1.png" style="width:40%; border:0;">
</p>

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/DCGAN_Interpolation.gif" style="border:0;">
</p>

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/Inter5.png" style="width:40%;border:0;">
</p>

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/Interpolation.gif" style="border:0;">
</p>

<h1 align = "center"> Kullback-Leibler Divergence </h1>

### KL Divergence of Two Distributions of Continuous Random Variable

$$ D\_{KL}(p \parallel q) = \underset{x \sim p(x)}{\int} p(x) \log \frac{p(x)}{q(x)} $$

### KL Divergence of Two Gaussians

For a $k$-dimensional Gaussian, the PDF $p(\mathbf{x})$ is given as:

$$ p(\mathbf{x}) = \frac{1}{(2 \pi)^{k/2} | \Sigma |^{1/2}} \exp \left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) $$

Let $p$ and $q$ be two Normal Distributions denoted as $\mathcal{N}(\boldsymbol{\mu}_p, \Sigma_p)$ and $\mathcal{N}(\boldsymbol{\mu}_q, \Sigma_q)$ respectively.

Then the KL Divergence between these two:

$$ D\_{KL}(p \parallel q) = \mathbb{E}\_p[ \log(p) - \log(q)] $$

$$ D\_{KL}(p \parallel q) = \frac{1}{2} [ \log \frac{| \Sigma_q |}{| \Sigma_p |} - k + (\boldsymbol{\mu_p} - \boldsymbol{\mu_q})^T \Sigma_q^{-1} (\boldsymbol{\mu_p} - \boldsymbol{\mu_q}) + tr \{ \Sigma_q^{-1} \Sigma_p \}] $$

In the scenario when $q$ is $\mathcal{N}(0, I)$, we get

$$ D\_{KL}(p \parallel q) = \frac{1}{2} [ \boldsymbol{\mu_p}^T \boldsymbol{\mu_p} + tr \{ \Sigma_p \} - k - \log |\Sigma_p| ] $$

<h1 align = "center"> Generative Adversarial Network </h1>

```math
\begin{equation}
\min_{\phi} \max_{\theta} V(G, D) = \underset{\textbf{x} \sim p{\text{data}}}{\mathbb{E}} [\log(D_{\theta}(\textbf{x}))] + \underset{\textbf{z} \sim p_z(\textbf{z})}{\mathbb{E}} [1 - \log(D_{\theta}(G_{\phi}(\textbf{z})))]
\end{equation}
```

<h1 align = "center">Variational AutoEncoders</h1>

We wish to achieve two goals:

1. **Learning Abstraction** $\to$ A hidden representation given the input $P(z | X)$ - this is achived by the **Encoder** $Q_{\theta}(z | X)$.
2. **Generation** $\to$ given some hidden representation using the **Decoder** $P_{\phi}(X | z)$.

For all these our aim to understand the joint distribution $P(X, z) = P(z) \cdot P(X | z)$. At inference we want given some $X$ (observed variable), finding out the most likely assignments of **latent variables** $z$ which would result in this observation.

$$ P(z | X) = \frac{P(X | z) \cdot P(z)}{P(X)} $$

But since $P(X) = \int P(X | z) \cdot P(z) dz = \int \int \dots \int P(X | z_1, z_2, \dots z_n) \cdot P(z_1, z_2, \dots z_n) dz_1 \cdot dz_2 \dots dz_n$ is **intractable**.

<p align="center">
<img src="https://github.com/guntas-13/SRIP2024/blob/master/Media/VAE.png" style="width:40%; border:0;">
</p>

Hence instead, we assume the posterior distribution $P(z | X)$ as $Q_{\theta}(z | X)$. Further assume that $Q_{\theta}(z | X)$ is a **Gaussian** whose parameters are determined by our neural network $\to$ **Encoder**.

$$ \boldsymbol{\mu}, \Sigma = g\_{\theta}(X) $$

Since

$$ D*{KL} (Q*{\theta}(z | X) \parallel P(z | X) ) = \int Q*{\theta}(z | X) \log \frac{Q*{\theta}(z | X)}{P(z | X)} \cdot dz $$

```math
\begin{equation}
= \underset{z \sim Q_{\theta}(z | X)}{\mathbb{E}} \left[ \log(Q_{\theta}(z | X)) - \log(P(z | X)) \right]
\end{equation}
```

```math
\begin{equation}
= \mathbb{E}_Q \left[ \log(Q_{\theta}(z | X)) - \log \left( \frac{P(X | z) \cdot P(z)}{P(X)} \right) \right]
\end{equation}
```

```math
\begin{equation}
= \mathbb{E}_Q \left[ \log(Q_{\theta}(z | X)) - \log(P(z)) \right] - \mathbb{E}_Q \left[ \log (P(X | z)) \right] + \log(P(X))
\end{equation}
```

Also since

```math
\begin{equation}
\mathbb{E}_Q [ \log(Q_{\theta}(z | X)) - \log(P(z)) ] = D_{KL} (Q_{\theta}(z | X) \parallel P(z) )
\end{equation}
```

so, the finally rearranging we may write

```math
\begin{equation}
\log(P(X)) = \color{red}{D_{KL} \left(Q_{\theta}(z | X) \parallel P(z | X) \right)} + \color{blue}{\mathbb{E}_Q \left[ \log (P(X | z)) \right] - D_{KL} \left(Q_{\theta}(z | X) \parallel P(z) \right)}
\end{equation}
```

since

```math
\begin{equation}
\color{red}{D_{KL} \left(Q_{\theta}(z | X) \parallel P(z | X) \right) \ge 0}
\end{equation}
```

```math
\begin{equation}
\color{blue}{\mathbb{E}_Q \left[ \log (P(X | z)) \right] - D_{KL} \left(Q_{\theta}(z | X) \parallel P(z) \right)} \le \log(P(X))
\end{equation}
```

And since the final task is maximising the log-likelihood of $P(X)$, hence it is equivalent to maximizing the $\color{blue}{\text{Blue Term}}$. So, the final objective is

```math
\begin{equation}
\color{green}{\mathcal{L}(\theta, \phi) = \max_{\theta, \phi} \left\{ \mathbb{E}_Q \left[ \log (P_{\phi}(X | z)) \right] - D_{KL} \left(Q_{\theta}(z | X) \parallel P(z) \right) \right \}}
\end{equation}
```

Now clearly all the terms are within our reach. To get the KL divergence, we make a forward pass through the **Encoder** to get $Q_{\theta}(z | X)$ and we know $P(z)$

$$ Q\_{\theta}(z | X) \sim \mathcal{N}(\boldsymbol{\mu_z}(X), \Sigma_z(X)) $$

$$ P(z) \sim \mathcal{N}(\mathbf{0}, I) $$

Now, in order for back propogation algorithm to work, we introduce the continuity in the sampling of $z$ by moving the sampling process to an input layer this is done first by sampling from a Standard Gaussian $\epsilon \sim \mathcal{N}(0, I)$ and then obtaing $z$ with the required $\boldsymbol{\mu_z}(X), \Sigma_z(X)$

$$ z = \boldsymbol{\mu_z}(X) + \Sigma_z(X) \times \epsilon $$

Hence, the randomness has been shifted to $\epsilon$ and not the $X$ or the parameters of the model.

<p align="center">
<img src="./Media/FinalVAE.png" style="width:40%; border:0;">
</p>

### Generation Part

After the model parameters are learned we **remove** the **encoder** and feed a $z \sim \mathcal{N}(0, I)$ to the decoder. The decoder will then predict $f_{\phi}(z)$ and we can draw an $X \sim \mathcal{N}(f_{\phi}(z), I)$.


## References
[1] L. Weng, “From autoencoder to beta-VAE,” lilianweng.github.io, 2018, Available: [https://lilianweng.github.io/posts/2018-08-12-vae/] <br>
[2] L. Weng, “What are diffusion models?” lilianweng.github.io, Jul. 2021, Available: [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/] <br>
[3] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models.” 2020. Available: [https://arxiv.org/abs/2006.11239]
