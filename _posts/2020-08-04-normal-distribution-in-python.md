---
layout: posts
shorttitle: Post
title: Normal distribution in Python
math: on
---

Working on theoretical statistics, all of my work in my PhD was done in R. But for production code in industry, for the sake of speed and easier maintenance, I have taken up to implement some of my ideas in Python even when the prototyping is done in R. I did not expect to be learning this much by just looking at a normal distribution.

## scipy.stats vs math

Suppose we want to compute the CDF of a standard normal. An R user like me who is used to `pnorm(z)` would probably write this

```python
from scipy.stats import norm

norm.cdf(z)
```

But another way is to use the `erf` function, given by

$$
\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} \,dt.
$$

We can then write

```python
from math import erf

def phi(z):
    return (1.0 + erf(z / sqrt(2.0))) / 2.0
```

In fact this is [the given example](https://docs.python.org/3/library/math.html#math.erf) in the Python documentations for the `math` library.

For my purpose, this function needs to be run many times, so speed is definitely important here. We run it for 10000 times:

```python
from timeit import default_timer
from scipy.stats import norm
from math import erf

def phi(z):
    return (1.0 + erf(z / 1.4142135623730951)) / 2.0

start = default_timer()
for i in range(10000):
    norm.cdf(1)
end = default_timer()
print("scipy took:", str(end - start))

start = default_timer()
for i in range(10000):
    phi(1)
end = default_timer()
print("math took:", str(end - start))

# scipy took: 1.3375982440000058
# math took: 0.003904457000004413
```

Computing using `erf` was a lot faster than using `scipy`, but that should not come as a surprise. Even in R, a loop without any vectorization is bound to be slow. After all, the strength of `scipy` is that it works well with `numpy` arrays. So let's vectorize it:

```python
import numpy

start = default_timer()
norm.cdf(numpy.repeat(1, 10000))
end = default_timer()
print("scipy took:", str(end - start))

# scipy took: 0.0018594119999875147
```

Too bad that in my use case, the CDF of the normal distribution has to be computed in a loop, so I will go with the `erf` route...

## erf vs erfc

...which brings us to another question. How accurate is computing the CDF based on erf? The part where we add `erf` to `1.0` means that anything that is close to or smaller than the machine epsilon will get erased. What if we do care about the magnitude of the CDF far in the tail of the normal? With the time constraint, we cannot really call `norm.logcdf` here.

```python
[phi(-z) for z in range(5, 15)]

# [2.8665157186802404e-07,
#  9.865876449133282e-10,
#  1.2798095916366492e-12,
#  6.106226635438361e-16,
#  0.0,
#  0.0,
#  0.0,
#  0.0,
#  0.0,
#  0.0]
```

We can instead use `erfc`, defined as

$$
\text{erfc}(z) = 1 - \text{erf}(z).
$$

Now we have

```python
from math import erfc

def phi(z):
    return erfc(-z / 1.4142135623730951) / 2.0

[phi(-z) for z in range(5, 15)]

# [2.866515718791945e-07,
#  9.865876450377014e-10,
#  1.2798125438858348e-12,
#  6.22096057427182e-16,
#  1.1285884059538425e-19,
#  7.619853024160593e-24,
#  1.9106595744986828e-28,
#  1.7764821120777016e-33,
#  6.117164399549921e-39,
#  7.793536819192798e-45]
```

It preserved many more digits. It is curious how the Python documentations chose to use this example for `erf` instead of `erfc`. To check if this is correct, I run in R:

```
pnorm(-(5:14))
```

which gave the exactly same numbers. Did R perhaps implement `pnorm` using `erfc` as well?
