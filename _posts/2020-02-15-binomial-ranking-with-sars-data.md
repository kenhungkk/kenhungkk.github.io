---
layout: posts
shorttitle: Post
title: Binomial ranking with SARS data
math: on
---

With the coronavirus spreading to many countries, Rebecca asked me a curious question: how does the US perform during SARS compared to other regions in terms of survival rate? While we can compute the survival rate of all infected regions and rank them accordingly, we are ignoring the sampling variability. For example, South Africa that has one case but also one death, does not necessarily perform worse than Indonesia where there were two cases but both patients survived.

## Setup

Of course there are many other factors, but we can consider an idealized model where the number of patients, $$n_i$$ in each country is predetermined, but the number of deaths, $$X_i$$ is random and comes from a binomial draw:

$$X_i \sim \text{Binomial}(n_i, p_i),$$

where $$p_i$$ represent the chance of a patient dying in region $$i$$. We would then want to provide simultaneous confidence intervals for the rank of region $$i$$, $$r_i$$, as defined in [Al Mohamad, Goeman, van Zwet (2018)](https://arxiv.org/abs/1812.05507):

$$r_i = 1 + \#\{j \ne i: p_j < p_i\}$$

There is an obvious Bayesian way to achieve this. By setting up a reasonable prior, we can perform posterior draws of $$p_i$$ and search for confidence intervals of ranks that covers $$(1 - \alpha)$$ of the posterior draws. Naturally this can be extended to an empirical Bayes way as well, as suggested in one of the comments in [this Cross Validated thread](https://stats.stackexchange.com/questions/157437/ranking-based-on-binomial-data-example-website-conversions). We want to focus on strict frequentist methods here.

## Data

I have never done data scraping, so I am glad that this led me to learn `rvest`. We read in the table from the [SARS page](https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome#Epidemiology) on Wikipedia. It looks like this:

```R
library(dplyr)
library(rvest)
library(tidyr)

data <- "https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome" %>%
  read_html %>%
  html_nodes(xpath = '//*[@id="mw-content-text"]/div/table[2]') %>%
  html_table(fill = TRUE)
data <- data[[1]] %>%
  setNames(c('region', 'cases', 'deaths', 'fatality', 'X1')) %>%
  select(region, cases, deaths) %>%
  filter(!grepl('total', tolower(region)), !grepl('\\^', region)) %>%
  mutate(
    region = trimws(gsub('\\[[[:print:]]\\]', '', region)),
    cases = as.numeric(gsub(',', '', cases)),
    deaths = as.numeric(gsub(',', '', deaths)),
    fatality = deaths / cases
  )
data
```

| region           | cases | deaths | fatality   |
|------------------|-------|--------|------------|
| China (mainland) | 5327  | 349    | 0.06551530 |
| Hong Kong        | 1755  | 299    | 0.17037037 |
| Taiwan           | 346   | 37     | 0.10693642 |
| Canada           | 251   | 43     | 0.17131474 |
| Singapore        | 238   | 33     | 0.13865546 |
| Vietnam          | 63    | 5      | 0.07936508 |
| United States    | 27    | 0      | 0.00000000 |
| Philippines      | 14    | 2      | 0.14285714 |
| ...              | ...   | ...    | ...        |

## Method 1: Simultaneous confidence intervals

One way is to construct simultaneous confidence intervals for each of the region, and "project" to figure out the ranks. We construct UMAU simultaneous confidence intervals:

{: .center}
![Simultaneous UMAU intervals for SARS fatality](images/binomial-ranking-sars-umau.png){:width="95%"}

These UMAU intervals are random by nature. Furthermore, the simultaneous coverage is achieved by Bonferroni correction here. A more fine-tuned analysis can be obtain by strategically distribute the type I error over the 29 regions, especially when some of these intervals are likely uninformative.

We can now compute the ranks for all parameters falling into the simultaneous confidence region:

| region           | cases | deaths | fatality   | ci.lo        | ci.hi      | rank.lo | rank.hi |
|------------------|-------|--------|------------|--------------|------------|---------|---------|
| China (mainland) | 5327  | 349    | 0.06551530 | 0.0553903386 | 0.07671217 | 1       | 26      |
| Hong Kong        | 1755  | 299    | 0.17037037 | 0.1432909366 | 0.19993250 | 2       | 29      |
| Taiwan           | 346   | 37     | 0.10693642 | 0.0619888081 | 0.16667006 | 1       | 29      |
| Canada           | 251   | 43     | 0.17131474 | 0.1042017481 | 0.25591583 | 2       | 29      |
| Singapore        | 238   | 33     | 0.13865546 | 0.0795036998 | 0.21678338 | 2       | 29      |
| Vietnam          | 63    | 5      | 0.07936508 | 0.0099142628 | 0.22994923 | 1       | 29      |
| United States    | 27    | 0      | 0.00000000 | 0.0000000000 | 0.18652527 | 1       | 29      |
| Philippines      | 14    | 2      | 0.14285714 | 0.0001578934 | 0.58405200 | 1       | 29      |
| ...              | ...   | ...    | ...        | ...          | ...        | ...     | ...     |

## Method 2: Simultaneous pairwise tests

Binomial distributions is an exponential family, so contrasts like $$p_i - p_j$$ are amenable to UMPU tests. We can perform $$\binom{29}{2}$$ pairwise tests, with Bonferroni correction, and draw conclusions about the ranks.

| region           | cases | deaths | fatality   | rank.lo | rank.hi |
|------------------|-------|--------|------------|---------|---------|
| China (mainland) | 5327  | 349    | 0.06551530 | 1       | 27      |
| Hong Kong        | 1755  | 299    | 0.17037037 | 2       | 29      |
| Taiwan           | 346   | 37     | 0.10693642 | 1       | 29      |
| Canada           | 251   | 43     | 0.17131474 | 2       | 29      |
| Singapore        | 238   | 33     | 0.13865546 | 1       | 29      |
| Vietnam          | 63    | 5      | 0.07936508 | 2       | 29      |
| United States    | 27    | 0      | 0.00000000 | 1       | 29      |
| Philippines      | 14    | 2      | 0.14285714 | 1       | 29      |
| ...              | ...   | ...    | ...        | ...     | ...     |

Here we are bounded to correct for all $$\binom{29}{2}$$ tests, instead of taking advantage of Tukey's HSD, which incurs a much smaller penalty for multiple testing. Asymptotically we can always think of the likelihood as if it came from a Gaussian distribution and use Tukey's HSD, but we most definitely are not in any reasonable asymptotic regime here.

## Thoughts

Are there better, strictly frequentist methods for computing these rank confidence intervals? This alone seems hard, but there is a natural, even harder generalization of this problem: suppose we have a joint distribution of exponential families where the base measure does not need to be the same

$$p_i(X_i; \theta_i) = h(x_i) \exp(\theta_i x_i - A_i(\theta_i)),$$

is there a powerful method for ranking $$\theta_i$$?
