Deep Learning Library (DLL)
===========================

DLL is a small library that aims to provide a C++ implementation of
Restricted Boltzmann Machine (RBM) and Deep Belief Network (DBN).

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

This library is completely header-only, there is no need to build it.

The folder **include** must be included with the **-I** option, as well as the
**etl/include** folder

However, this library makes extensive use of C++11 and C++14, therefore, a
recent compiler is necessary to use it. This library is tested on CLang 3.4.1.
It should compile on g++ 4.9.1, but it does not because G++ refuses to call
static functions inside lambdas.  This will eventually be fixed out.

This has never been tested on Windows. While it should compile on Mingw, I don't
expect Visual Studio to be able to compile it for now. If you have problems
compiling this library, I'd be glad to help, but I do not guarantee that this
will work on other compilers.

License
-------

This library is distributed under the terms of the MIT license, see `LICENSE`
file for details.
