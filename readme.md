# A Stochastic Proximal Method for Non-smooth Regularized Finite Sum Optimization

## Description

SR2 is an optimizer that trains deep neural networks with nonsmooth regularization to retrieve a sparse and efficient sub-structure by minimizing a finite sum problem expressed as the sum of a loss function and a nonsmooth nonconvex regularizer:

    F(x) = l(x) + R(x) 

Supported regularizers are $\ell_0$ and $\ell_1$.

## Prerequisits 
 - Numpy
 - Pytorch
 - PyHessian [https://github.com/amirgholami/PyHessian]
