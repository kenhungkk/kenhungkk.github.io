---
layout: posts
shorttitle: Post
title: Playing with Bayes and RStan
math: on
---

I did not really much statistical training in my undergrad days, and my knowledge of statistics is pretty much confined to whatever grad level statistics classes Berkeley offered --- 99% of those was frequentist --- so I lack the Bayesian exposure that most statistics undergrad would have received. So when something slightly Bayesian (does empirical Bayes count?) showed up, I decided to teach myself using Gelman et al.'s Bayesian Data Analysis (BDA).

It is hard to learn something new without any examples, and I happen to stumble upon this tweet:

<blockquote class="twitter-tweet tw-align-center"><p lang="en" dir="ltr">Trend line can be misleading, and there are good years too. But certainly it looks like bad years had become worse. Just curious, where can I find the data itself for educational purposes?</p>&mdash; Kenneth Hung (@kenhungkk) <a href="https://twitter.com/kenhungkk/status/1302973968748478464?ref_src=twsrc%5Etfw">September 7, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

The data itself is [here](https://www.fire.ca.gov/media/11397/fires-acres-all-agencies-thru-2018.pdf) and I got to [learn how to handle PDFs](https://blog.az.sg/posts/reading-pdfs-in-r/) with `pdftools` as well.

```r
library(pdftools)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(rstan)
library(foreach)
library(doParallel)

theme_set(theme_minimal())
options(mc.cores = parallel::detectCores())
registerDoParallel(cores = parallel::detectCores())

raw.text <- pdf_text(
  'https://www.fire.ca.gov/media/11397/fires-acres-all-agencies-thru-2018.pdf'
) %>%
  str_split("\n") %>%
  unlist()

data <- raw.text[4:35] %>%
  as_tibble(.name_repair = "unique") %>%
  mutate(
    full.year = str_sub(value, end = 6), acres = str_sub(value, start = 132)
  ) %>%
  select(-value) %>%
  mutate_all(str_trim) %>%
  mutate_all(str_replace_all, ",", "") %>%
  mutate_all(as.numeric) %>%
  mutate(year = full.year - min(full.year))
N <- nrow(data)

ggplot(data) + geom_line(aes(x = full.year, y = acres))
```

{: .center}
![Acres burnt as recorded by Cal Fire](images/cal-fire-time-series.png)

Like I said in the tweet, a linear fit seems off.

Using the default priors, I ran a Bayesian linear regression. I have to say, seeing the chains running and utilizing my new laptop's computation power was very exciting.

```r
model.stan <- "
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
}
"

fit <- stan(
  model_code = model.stan,
  data = list(N = N, x = data$year, y = data$acres),
  iter = 5000
)

summary(fit)$summary

#              mean      se_mean           sd        2.5%         25%         50%         75%       97.5%
# alpha 238814.9749 2.567240e+03 1.555590e+05 -69867.4091 136989.2119 240085.6816 339548.6771 549296.6163
# beta   25395.4792 1.447920e+02 8.690706e+03   7883.9314  19805.8538  25291.1804  30983.0647  42960.0410
# sigma 446924.4876 9.068805e+02 6.112602e+04 345835.7543 404007.3506 440055.5536 482446.6112 587458.8149
# lp__    -418.5239 2.259724e-02 1.301506e+00   -421.8609   -419.1331   -418.1785   -417.5646   -417.0523
#          n_eff     Rhat
# alpha 3671.615 1.001180
# beta  3602.643 1.001261
# sigma 4543.098 1.000833
# lp__  3317.277 0.999997
```

Since all Bayesians do is do posterior draws, I don't find it hard to understand the result. But what matters the most to me is that the data is fit well. As BDA would suggest, I should do a posterior predictive check, specifically something that would demonstrate my suspicion that linear model isn't a good fit. By looking at the time series, I would guess that there are fewer positive residuals than negative ones. So I used the proportion of positive residuals after OLS as the test statistic. Embarrassingly it took me a long time to realize how to do a posterior predictive check for regression, the most basic example in Chapter 14 surprisingly did not emphasize this part.

```r
post <- extract(fit)
post.pred.stat <- foreach(
  alpha = post$alpha,
  beta = post$beta,
  sigma = post$sigma,
  .combine = "c"
) %dopar% {
  y.rep <- alpha + beta * data$year + sigma * rnorm(N)
  residual.rep <- residuals(lm(y.rep ~ data$year))
  mean(residual.rep > 0)
}

residual <- residuals(lm(acres ~ year, data = data))
obs.stat <- mean(residual > 0)

ggplot() +
  geom_histogram(
    aes(x = post.pred.stat), alpha = 0.5, binwidth = 1 / nrow(data)
  ) +
  geom_vline(xintercept = obs.stat)
```

{: .center}
![Linear model residual assymetry](images/cal-fire-linear-symmetry.png)

It looks like we have way too few positive residuals and I should probably use a log-linear model.

```r
model.stan <- "
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
}
"

fit <- stan(
  model_code = model.stan,
  data = list(N = N, x = data$year, y = log(data$acres)),
  iter = 5000
)

summary(fit)$summary

#              mean     se_mean         sd         2.5%        25%         50%         75%       97.5%    n_eff
# alpha 12.41139306 0.004204351 0.26721568  11.88718996 12.2342716 12.41242466 12.58896869 12.92811368 4039.484
# beta   0.04208909 0.000230521 0.01477833   0.01344978  0.0321454  0.04202818  0.05209784  0.07101945 4109.885
# sigma  0.77109927 0.001601357 0.10725044   0.59443148  0.6970235  0.75938712  0.83172974  1.01526800 4485.616
# lp__  -7.17092179 0.022112438 1.29931384 -10.50676341 -7.7550080 -6.84540958 -6.23299112 -5.69367149 3452.668
#           Rhat
# alpha 1.000576
# beta  1.000951
# sigma 1.000233
# lp__  1.001881
```

And we perform the same check to see if the residuals are symmetric.

```r
post <- extract(fit)
post.pred.stat <- foreach(
  alpha = post$alpha,
  beta = post$beta,
  sigma = post$sigma,
  .combine = "c"
) %dopar% {
  y.rep <- alpha + beta * data$year + sigma * rnorm(N)
  residual.rep <- residuals(lm(y.rep ~ data$year))
  mean(residual.rep > 0)
}

residual <- residuals(lm(log(acres) ~ year, data = data))
obs.stat <- mean(residual > 0)

ggplot() +
  geom_histogram(
    aes(x = post.pred.stat), alpha = 0.5, binwidth = 1 / nrow(data)
  ) +
  geom_vline(xintercept = obs.stat)
```

{: .center}
![Loglinear model residual symmetry](images/cal-fire-loglinear-symmetry.png)

Much better! Of course the model is not going to be correct, but we just need to keep checking for statistics that we care about. One idea I had is the number of times a new record is set. From the time series, it is 5 --- we will count the very first year, not that it really matters. I thought this may be revealing if there is a lot of autocorrelation in the time series --- for example, the more acres are burnt the previous year, the less there is to burn the year after.

```r
post <- extract(fit)
post.pred.stat <- foreach(
  alpha = post$alpha,
  beta = post$beta,
  sigma = post$sigma,
  .combine = "c"
) %dopar% {
  y.rep <- alpha + beta * data$year + sigma * rnorm(N)
  sum(y.rep == cummax(y.rep))
}

obs.stat <- sum(data$acres == cummax(data$acres))

ggplot() +
  geom_histogram(
    aes(x = post.pred.stat), alpha = 0.5, binwidth = 1
  ) +
  geom_vline(xintercept = obs.stat)
```

{: .center}
![Loglinear model new records](images/cal-fire-loglinear-record.png)

Not too bad! For both model it looks like the slope is positive. There are many other data that would have been relevant to this analysis, such as the rainfall the year before and other climate data. There are also more sophisticated things such as Bayesian ARIMA that I could do (but I don't know how), but hey, there are only 32 points in this dataset.
