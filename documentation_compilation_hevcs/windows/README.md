# Compilation of HEVC and two versions of HEVC including the neural networks on Windows

## Compiling HEVC
Open the solution file "hevc\hm_16_15_regular\build\HM_vc2015.sln" using Visual Studio and compile.

## Compiling the two versions of HEVC including the neural networks by linking to static Tensorflow libraries
In the versions of HEVC including the neural networks for intra prediction, the neural
networks will be instantiated and run using C++ Tensorflow libraries. In HEVC, all libraries
are static. That is why the Tensorflow libraries have to be static too.

Google recommends to compile Tensorflow from source via Bazel, see [BazelWebPage](https://bazel.build/).
But, Bazel requires lot of RAM and dependencies. Alternatively, the compilation described
below uses CMake.

### Compiling static Tensorflow libraries via CMake
1. Get the Tensorflow repository. It is recommended to get Tensorflow 1.9.0, see
[TF-1.9.0WebPage](https://github.com/tensorflow/tensorflow/releases/tag/v1.9.0), as
the CMake support in this version is up-to-date and the protocol below was tested using
Tensorflow 1.9.0. To avoid modifying paths later on when linking the built static
Tensorflow libraries to HEVC executables, it is recommended to put the Tensorflow
repository into the directory containing the root directory of the current project
named "context_adaptive_neural_network_based_prediction".
2. Make sure that the CMake executable and the Visual Studio executable
named "msbuild.exe" are in your %PATH%.
3. Put the files "run_static_library_cmake.bat" and "run_static_library_build.bat"
into the root directory of the Tensorflow repository.
4. Follow the instructions in the file "note_modifications_cmake.txt" to change
the Tensorflow CMake files such that static Tensorflow libraries will be built
instead of dynamic libraries. To apply these modifications faster, you can also
look at the directory "cmake_with_modifications_tensorflow_1.9.0". It contains
the file "CMakeLists.txt" and the other CMake files with the required modifications
in the case of Tensorflow 1.9.0.
5. Run "run_static_library_cmake.bat" to create the Tensorflow Visual Studio projects and run "run_static_library_build.bat" to compile static Tensorflow libraries.
   ```sh
   cd ..\tensorflow-1.9.0
   run_static_library_cmake.bat
   run_static_library_build.bat
   cd ..\context_adaptive_neural_network_based_prediction
   ```

### Compiling the two versions of HEVC including the neural networks
The two solution files for compilation use two macros defined in the property
sheet "hevc\hm_common\properties_substitution_switch.props". The first macro
is the absolute path to the directory containing "python.exe". The second macro
is the path to the root of the Tensorflow repository, relatively to the solution
file directory.

1. Modify the two macros in the property sheet "hevc\hm_common\properties_substitution_switch.props"
if needed.
2. Open the solution file "hevc\hm_16_15_substitution\build\HM_vc2015.sln" using Visual Studio.
Make sure that the configuration is "Release" and the platform is "x64".
3. Compile "hevc\hm_16_15_substitution\build\HM_vc2015.sln".
4. Open and compile "hevc\hm_16_15_switch\build\HM_vc2015.sln", as in (1) and (2).

## Useful links
  * https://medium.com/@arnaldog12/how-to-build-tensorflow-on-windows-with-mt-42a8e4bea7e7
  * https://joe-antognini.github.io/machine-learning/build-windows-tf


