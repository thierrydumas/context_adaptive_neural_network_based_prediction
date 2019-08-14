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
 
## Quick start: reproducing the main results of the paper
1. Creation of the Kodak test set containing 24 YCbCr images.
   ```sh
   python creating_kodak.py
   ```
2. Creation of the BSDS test set containing 100 YCbCr images. Below, `/path/to/directory_0`
   is the path to the directory storing the archive of the BSDS dataset the script `creating_bsds.py`
   downloads from [BSDSWebPage](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).
   `/path/to/directory_1` is the path to the directory storing the RGB images extracted from
   this downloaded archive.
   ```sh
   python creating_bsds.py /path/to/directory_1 --path_to_directory_tar=/path/to/directory_0
   ```
3. Reproducing the results in Tables I, II, III, IV, and V. Note that the pretrained models of the
   neural networks for predicting blocks of size 4x4, 8x8, 16x16, and 32x32 are provided in the
   code. However, the pretrained models of the neural networks for predicting blocks of size 64x64
   are not given as their size is too large, which prevents from pushing them remotely. Please, either
   write to dumas--thierry@hotmail.fr to get the missing pretrained models or retrain them.
   ```sh
   python comparing_pnn_ipfcns_hevc_best_mode.py --all
   ```
4. Freezing the graphs and the parameters of the neural networks to use them inside HEVC/H.265 in (5).
     ```sh
	 python freezing_graph_pnn.py --all
	 ```
5. Reproducing the results in Tables IX and X. Below, `/path/to/directory/data` is the path
   to the directory storing the YUV sequence to be encoded and decoded via  HEVC/H.265 and two
   variants of HEVC/H.265 using the neural networks for intra predicton. `prefix` is the prefix of
   the name of this YUV sequence, e.g. "D_BasketballPass", "B_Kimono", "C_BasketballDrill" or "Bus".
   ```sh
   python comparing_rate_distortion.py ycbcr --path_to_directory_data=/path/to/directory/data --prefix_filename=prefix
   ```


