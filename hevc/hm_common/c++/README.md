# Tools for the two versions of HEVC/H.265 with the neural networks for intra prediction

This C++ code contains tools for integrating the neural networks for intra prediction
into HEVC/H.265.

## Compiling the tools
  * In Windows,
    * 0pen the solution file "hevc\hm_common\c++\build\integration.sln".
	* Choose the "Release" configuration and the "x64" platform.
	* Compile.
  * In Linux,
  ```sh
  cd hevc/hm_common/c++/build/linux
  make
  cd ../../../../../
  ```

## Run the tests of the tools
The tests must be run from the root directory of the current project. This is because
many paths in these tests are defined relatively to this directory.

For instance, in Windows,
```sh
hevc\hm_common\c++\bin\vc2015\x64\Release\integration.exe extract_context_portions
```
For instance, in Linux,
```sh
./hevc/hm_common/c++/bin/executable extract_context_portions
```


