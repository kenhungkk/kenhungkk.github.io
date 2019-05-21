---
layout: posts
shorttitle: Post
title: ℓ1-Penalized Likelihood Asymptotics
math: on
---
Following a paper by [Lee et al. (2013)](https://arxiv.org/abs/1311.6238) on the correcting for the selection bias after lasso-based selection, a natural progression is to consider general penalized likelihood selections. In GLM, we are at least provided with a sufficient statistics, but this would not be the case in a more general likelihood setting, rendering the description of the selection event a lot more blurry.

Most specifically, the set up is as follows: we have a matrix $$X$$ consisting of $$n$$ row of covariates, $$Y$$ a vector conssiting of all the responses. We assume the model, given by the log-likelihood below,

$$\sum_{i=1}^n \ell(\theta_n; Y_n, X_n).$$

Here we will not move into high-dimensional regime and thus assumes $$\theta_n$$ to have dimension $$d$$. Subsequently, we perform selection based on maximizing

$$\sum_{i=1}^n \ell(\theta_n; Y_n, X_n) - \lambda_n \|\theta_n\|_1,$$

with some tuning parameter $$\lambda_n$$.

Around three weeks ago, Will Fithian and I came up with a way to tackle this problem --- together with a non-exhaustive, non-optimized list of conditions needed. Unfortunately shortly after Will discovered a paper by [Taylor and Tibshirani (2017)](https://arxiv.org/abs/1602.07358) that arrived at an almost identical solution. While our result might no longer be groundbreaking, we hope that this post will provide a different perspective from Taylor and Tibshirani (2017), and assist anyone who happens to also be reading Taylor and Tibshirani (2017).

The problem has two main hurdles:
- approximating the selection event in a reasonable yet theoretically valid manner;
- choosing a test statistic with a nice asymptotic distribution.

We can make both decisions at once by considering the selection event. In GLM, with a sufficient statistic, the selection event will always be measurable with respect to this sufficient statistic. This measurability requirement results in fuzzy edges if we plot out the selection event based on a non-sufficient statistic.

We don't have this sufficient statistic anymore in a general likelihood setting. Conventionally, both the score at a fixed parameter and the MLE are thought of as 'asymptotically sufficient' without a proper definition. Since we are looking into asymptotics anyways, these two statistic seems perfect for our use. A bonus is that their asymptotic distributions are well known.

Following classical asymptotic analysis as explained in van der Vaart (1998), we will assume that $$\theta_n = \theta_0 + \beta / \sqrt{n}$$ and thus converges to a $$\theta_0$$ that lies in the null hypothesis $$\Theta_0$$. Other possible asymptotic regimes includes modifying the lasso minimization problem into a 'non-centered' lasso problem

$$\sum_{i=1}^n \ell(\theta_n; Y_n, X_n) - \lambda_n \|\theta_n - c_n\|_1,$$

but as it turns out the asymptotics will work out to the same solution anyways. For the lasso selection to not go trivial (always selecting certain variables, always not selecting certain variables, always making the correct selection), we also need to scale $$\lambda_n$$ as $$\lambda_n = \lambda \sqrt{n}$$.

If we take the subgradient of the objective, normalized by $$1 / \sqrt{n}$$, with respect to $$\theta_n$$, we get something like

$$\frac{1}{\sqrt{n}} \sum_{i=1}^n \nabla \ell(\theta_n; Y_n, X_n) - \lambda s_n,$$

where $$s_n$$ is the subgradient of the $$\ell_1$$-norm. This is the crucial step in Lee et al. (2013). For a the same set of variables selected and the same signs assigned, $$\lambda s_n$$ is a determined set. So what's left is to relate the normalized score to the sufficient statistic.

In the asymptotic regime, the asymptotic sufficiency of score and the MLE means we can determine all the likelihood ratio, or equivalently, the entire sore function. From here we can approximate the score as a linear function at $$0$$ as

$$\frac{1}{\sqrt{n}} \sum_{i=1}^n \nabla \ell(\theta_n; Y_n, X_n) \approx \left[\frac{1}{n} \sum_{i=1}^n \nabla^2 \ell(0; Y_n, X_n)\right] \beta,$$

or as a linear function based at the MLE $$\hat{\theta}_n$$ (and hence $$\hat{\beta}$$),

$$\frac{1}{\sqrt{n}} \sum_{i=1}^n \nabla \ell(\theta_n; Y_n, X_n) \approx \left[\frac{1}{n} \sum_{i=1}^n \nabla^2 \ell(\hat{\theta}_n; Y_n, X_n)\right] (\beta - \hat{\beta}).$$

We cannot however approximate this as a linear function at other points, such as the MLE restricted to the null hypothesis $$\Theta_0$$, as it reduces the degree of freedom.

How do we choose between these two approximation? In finite sample, the 'data' might not lie close to $$0$$, rendering the first approximation ill-motivated. The second one has an appeal that it moves with the data and tends to approximate the score function better locally near the MLE.

To be more concrete, we can have a look at this in practice. We generated 1000 samples of 100 points from a logistic model and ran `glmnet` on each of the 100 samples. The unrestricted MLE is used as the statistic and plotted below. Colors follow the signs and the variables selected.

{: .center}
![Selection event simulation](files/selection-event-simulation.png)

We then look specifically at the sample marked with `x`. The approximating the selection event based on the score at zero will approximate the event much better around the origin, but we also care much less about this scenerio. The 'high stake' scanerio is when the statistic is close the the boundaries --- and in these cases we would want the selection event to be approximated better for that section of the boundary. The MLE thus appeals to this.

The Hessian of the log-likelihood has to be approximated as well. The selection event given by the true Hessian is given as dashed lines above, while the estimated Hessian is given as solid lines. Notice that while the approximation on the left edge of the red region is not done well, the approximation is done well in the bottom edge, which is more important to us. Also notice that the estimated Hessian performs fairly well.

Finally, how is these approximations linked to that of Taylor and Tibshirani (2017)? They used the lasso estimate with one extra Newton step as their test statistic. Assuming the log-likelihood behaves sufficiently quadratic, this is the same as using the MLE. We admit that their approach probably has a slight edge, an MLE would require solving a whole new approximation problem, while a one-extra-Newton-step lasso estimate is extremely easy to compute. In application, we believe these two methods should perform similarly.