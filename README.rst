Deep Learning Library (DLL) 1.1
===============================

|logo|    |coverage| |jenkins| |license|

.. |logo| image:: logo_small.png
.. |coverage| image:: https://img.shields.io/sonar/https/sonar.baptiste-wicht.ch/dll/coverage.svg
.. |jenkins| image:: https://img.shields.io/jenkins/s/https/jenkins.baptiste-wicht.ch/dll.svg
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg

DLL is a library that aims to provide a C++ implementation of Restricted
Boltzmann Machine (RBM) and Deep Belief Network (DBN) and their convolution
versions as well. It also has support for some more standard neural networks.

Features
--------

* **Restricted Boltzmann Machine**

  * Various units: Stochastic binary, Gaussian, Softmax and nRLU units
  * Contrastive Divergence and Persistence Contrastive Divergence

    * CD-1 learning by default

  * Momentum
  * Weight decay
  * Sparsity target
  * Train as Denoising autoencoder

* **Convolutional Restricted Boltzmann Machine**

  * Standard version
  * Version with Probabilistic Max Pooling (Honglak Lee)
  * Binary and Gaussian visible units
  * Binary and ReLU hidden units for the standard version
  * Binary hidden units for the Probabilistic Max Pooling version
  * Training with CD-k or PCD-k (only for standard version)
  * Momentum, Weight Decay, Sparsity Target
  * Train as Denoising autoencoder

* **Deep Belief Network**

  * Pretraining with RBMs
  * Fine tuning with Conjugate Gradient
  * Fine tuning with Stochastic Gradient Descent
  * Classification with SVM (libsvm)

* **Convolutional Deep Belief Network**

  * Pretraining with CRBMs
  * Classification with SVM (libsvm)

* Input data

  * Input data can be either in containers or in iterators

    * Even if iterators are supported for SVM classifier, libsvm will move all
      the data in memory structure.

Building
--------

Note: When you clone the library, you need to clone the sub modules as well,
using the --recursive option.

The folder **include** must be included with the **-I** option, as well as the
**etl/include** folder.

This library is completely header-only, there is no need to build it.

However, this library makes extensive use of C++20 and C++23, therefore,
a recent compiler is necessary to use it. Currently, this library is only tested
with g++ 13.

If for some reasons, it should not work on one of the supported compilers,
contact me and I'll fix it. It should work fine on recent versions of clang.

This has never been tested on Windows. While it should compile on Mingw, I don't
expect Visual Studio to be able to compile it for now, although recent versions of VS sound
promising. If you have problems compiling this library, I'd be glad to help, but
cannot guarantee that this will work on other compilers.

If you want to use GPU, you should use CUDA 12 or superior and CUDNN 8 or
superior. If you got issues with different
versions of CUDA and CUDNN, please open an issue on Github.

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE`
file for details.
