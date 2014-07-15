Deep Learning Library (DLL)
===========================

DLL is a small library aims to provide a C++ implementation of Restricted
Boltzmann Machine (RBM) and Deep Belief Network (DBN).

Features
--------

* **Restricted Boltzmann Machine**

   * Various units: Stochastic binary, Gaussian, Softmax and nRLU units
   * Contrastive Divergence and Persistence Contrastive Divergence
      * CD-1 learning by default
   * Momentum
   * Weight decay
   * Sparsity target

* **Convolutional Restricted Boltzmann Machine**

  * Standard version
  * Version with Probabilistic Max Pooling (Honglak Lee)
  * Binary and Gaussian visible units
  * Binary and ReLU hidden units for the standard version
  * Binary hidden units for the Probabilistic Max Pooling version
  * Training with CD-k or PCD-k (only for standard version)
  * Momentum, Weight Decay, Sparsity Target

* **Deep Belief Network**

   * Pretraining with RBMs
   * Fine tuning with Conjugate Gradient

In development
--------------

conv_rbm is still in heavy development and should not be used unless you intend
to fix it ;)

Building
--------

This library is completely header-only, there is no need to build it.

The folder **include** must be included with the **-I** option, as well as the
**etl/include** folder

However, this library makes extensive use of C++11 and C++1y, therefore, a
recent compiler is necessary to use it.
This library has only been tested on CLang 3.4, but should work on latest
version of GCC too.

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE`
file for details.
