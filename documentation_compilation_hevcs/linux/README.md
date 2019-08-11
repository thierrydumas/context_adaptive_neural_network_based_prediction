# Compilation of HEVC and two versions of HEVC including the neural networks on Linux

## Compiling HEVC
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
Simply follow the instructions provided by the Tensorflow Makefile contribution, see
[MakefileContributionWebPage](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile).
Then, the directory "tensorflow/contrib/makefile/gen/lib" (the path being given from the root directory
of the Tensorflow repository) should contain the static Tensorflow library "libtensorflow-core.a".

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


