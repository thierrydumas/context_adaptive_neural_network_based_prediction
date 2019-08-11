#ifndef INTERFACE_C_PYTHON_H
#define INTERFACE_C_PYTHON_H

/*
"Python.h" implies inclusion of the following standard
headers: <stdio.h>, <string.h>, <errno.h>, <limits.h>,
<assert.h> and <stdlib.h>.
*/
#ifdef _WIN32
    #include "Python.h"
#else
    #include "python2.7/Python.h"
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/*
"Python.h" must be included before any standard
header is included.
*/
#include <string>

/** @brief Appends `path_to_additional_directory` to the Python sys path list.
  *
  * @param path_to_additional_directory Path to be appended to the Python sys path list.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int append_sys_path(const std::string& path_to_additional_directory);

/** @brief Gets a function in a file ".py". Returns a NEW REFERENCE.
  *
  * @param name_file Name of the file ".py" without the extension ".py".
  * @param name_function Name of the function.
  * @return Pointer to the function. It is NULL if an error occurs.
  *
  */
PyObject* get_callable(const std::string& name_file,
                       const std::string& name_function);

/** @brief Loads a Python object via pickle. Returns a NEW REFERENCE.
  *
  * @param python_function Pointer to the Python function that loads the Python object.
  * @param path_to_file Path to the file ".pkl" containing the Python object.
  * @return Pointer to the loaded Python object. It is NULL if an error occurs.
  *
  */
PyObject* load_via_pickle(PyObject* python_function,
                          const std::string& path_to_file);

#endif


