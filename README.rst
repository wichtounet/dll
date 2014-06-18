dbn
===

This library aims to provide a C++ implementation of Restricted Boltzmann
Machine (RBM) and Deep Belief Network (DBN).

Features
--------

* **Restricted Boltzmann Machine**

   * Various units: Stochastic binary, Gaussian, Softmax and nRLU units
   * CD1 Learning
   * Momentum and weight decay

* **Deep Belief Network**

   * Pretraining with RBMs
   * Fine tuning with Conjugate Gradient

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE` file for details.

Building
--------

This library is completely header-only, there is no need to build it.

However, this library makes extensive use of C++11 and C++1y, therefore, a recent compiler is necessary to use it.
This library has only been tested on CLang 3.4.
