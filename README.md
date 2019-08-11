# Context-adaptive neural network based prediction for image compression

This repository is a Tensorflow implementation of the paper "Context-adaptive neural network based prediction for image compression", *TIP*, 2019.

[TIP 2019 paper](https://arxiv.org/pdf/1807.06244.pdf) | [Project page with visualizations](https://www.irisa.fr/temics/demos/prediction_neural_network/PredictionNeuralNetwork.htm)

The code is tested on Linux and Windows.

## Prerequisites
  * Python (the code was tested using Python 2.7.9 and Python 3.5.4)
  * numpy (version >= 1.11.0)
  * tensorflow (optional GPU support), see [TensorflowInstallationWebPage](https://www.tensorflow.org/install/) (the code was tested using Tensorflow 1.4.2 and Tensorflow 1.5.1)
  * cython
  * matplotlib
  * pillow
  * scipy
  * six

## Cloning the code
Clone this repository into the current directory.
```sh
git clone https://github.com/thierrydumas/context_adaptive_neural_network_based_prediction.git
cd context_adaptive_neural_network_based_prediction
```

## Compilation
1. Compilation of the C++ code reproducing the HEVC/H.265 intra prediction modes via Cython.
```sh
cd hevc/intraprediction
python setup.py build_ext --inplace
cd ../../
```
2. Compilation of HEVC/H.265 and two versions of HEVC/H.265 including the neural networks for intra prediction.
    * For Linux, refer to the documentation at "documentation_compilation_hevcs/linux/README.md".
    * For Windows, refer to the documentation at "documentation_compilation_hevcs/windows/README.md".

Unfinished ...

