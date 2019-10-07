# Compilation of HEVC and two versions of HEVC including the neural networks on Linux

## Compiling HEVC
For details on the compilation of HEVC, see [HEVCSoftwareWebPage](https://hevc.hhi.fraunhofer.de/).
```sh
cd hevc/hm_16_15_regular/build/linux
make
cd ../../../../
```

## Compiling the two versions of HEVC including the neural networks by linking to a static Tensorflow library
In the versions of HEVC including the neural networks for intra prediction, the neural networks will
be instantiated and run using C++ Tensorflow libraries. In HEVC, all libraries are static. That is why
the Tensorflow library has to be static too.

Google recommends to compile Tensorflow from source via Bazel, see [BazelWebPage](https://bazel.build/).
But, Bazel requires lot of RAM and dependencies. Alternatively, the compilation described below uses Make.

### Compiling a static Tensorflow library
1. Get the Tensorflow repository. It is recommended to get Tensorflow 1.9.0, see
[TF-1.9.0WebPage](https://github.com/tensorflow/tensorflow/releases/tag/v1.9.0), as
the Make support in this version is stable and the protocol below was tested using
Tensorflow 1.9.0. To avoid modifying paths later on when linking the built static
Tensorflow libraries to HEVC executables, it is recommended to put the Tensorflow
repository into the directory containing the root directory of the current project
named "context_adaptive_neural_network_based_prediction".
2. Follow the instructions given by "tensorflow/contrib/make/README.md", the file
path being relative to the root directory of the Tensorflow repository. Then, the
directory "tensorflow/contrib/makefile/gen/lib"  should contain the static Tensorflow
library "libtensorflow-core.a".

### Compiling the two versions of HEVC including the neural networks
The Makefiles for compilation use a macro defined in the file "hevc/hm_common/common.mk".
This macro is the path to the root of the Tensorflow repository, relatively to the
directory "hevc/hm_common/".

1. Modify the macro in the file "hevc/hm_common/common.mk" if needed.
2. Compile the two versions of HEVC including the neural networks via Makefiles.
   ```sh
   cd hevc/hm_16_15_substitution/build/linux
   make
   cd ../../../hm_16_15_switch/build/linux
   make
   cd ../../../../
   ```


