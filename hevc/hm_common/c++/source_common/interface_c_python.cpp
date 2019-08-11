#include "interface_c_python.h"

int append_sys_path(const std::string& path_to_additional_directory)
{
    /*
    All pointers to Python objects are
    declared below.
    */
    PyObject* python_path_sys(NULL);
    PyObject* python_path_to_additional_directory(NULL);
    
    /*
    `PySys_GetObject` returns a borrowed reference. If "path"
    from the module named "sys" does not exist, `PySys_GetObject`
    returns NULL, without setting an exception.
    */
    std::string string_path("path");
    python_path_sys = PySys_GetObject(const_cast<char*>(string_path.c_str()));
    if (!python_path_sys)
    {
        fprintf(stderr, "`PySys_GetObject` fails.\n");
        return -1;
    }
    
    // `PyString_FromString` returns a new reference.
    python_path_to_additional_directory = PyString_FromString(path_to_additional_directory.c_str());
    if (!python_path_to_additional_directory)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyString_FromString` fails but the error indicator is not set.\n");
        }
        return -1;
    }
    
    /*
    `PyList_Insert` returns 0 if it is successful.
    It returns -1 and sets an exception if it is
    unsuccessful. `PyList_Insert` does not steal
    a reference.
    */
    const int error_code(PyList_Insert(python_path_sys, 0, python_path_to_additional_directory));
    Py_DECREF(python_path_to_additional_directory);
    if (error_code < 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyList_Insert` fails but the error indicator is not set.\n");
        }
        return -1;
    }
    return 0;
}

PyObject* get_callable(const std::string& name_file,
                       const std::string& name_function)
{
    /*
    All pointers to Python objects are
    declared below.
    */
    PyObject* python_name_file(NULL);
    PyObject* python_module(NULL);
    PyObject* python_function(NULL);
    
    // `PyString_FromString` returns a new reference.
    python_name_file = PyString_FromString(name_file.c_str());
    if (!python_name_file)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyString_FromString` fails but the error indicator is not set.\n");
        }
        return NULL;
    }
    
    // `PyImport_Import` returns a new reference.
    python_module = PyImport_Import(python_name_file);
    Py_DECREF(python_name_file);
    if (!python_module)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyImport_Import` fails but the error indicator is not set.\n");
        }
        return NULL;
    }
    
    // `PyObject_GetAttrString` returns a new reference.
    python_function = PyObject_GetAttrString(python_module, name_function.c_str());
    Py_DECREF(python_module);
    if (!python_function)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyObject_GetAttrString` fails but the error indicator is not set.\n");
        }
        return NULL;
    }
    
    /*
    `PyCallable_Check` returns 1 if its argument is
    callable. It returns 0 otherwise. The function
    always succeeds.
    */
    const int error_code(PyCallable_Check(python_function));
    if (!error_code)
    {
        fprintf(stderr, "The object is not callable.\n");
        Py_DECREF(python_function);
        return NULL;
    }
    return python_function;
}

PyObject* load_via_pickle(PyObject* python_function,
                          const std::string& path_to_file)
{
    /*
    All the pointers to Python objects are
    declared below.
    */
    PyObject* python_path_to_file(NULL);
    PyObject* python_loaded_object(NULL);
    
    if (!python_function)
    {
        fprintf(stderr, "`python_function` is NULL.\n");
        return NULL;
    }
    
    // `PyString_FromString` returns a new reference.
    python_path_to_file = PyString_FromString(path_to_file.c_str());
    if (!python_path_to_file)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyString_FromString` fails but the error indicator is not set.\n");
        }
        return NULL;
    }
    
    // `PyObject_CallFunctionObjArgs` returns a new reference.
    python_loaded_object = PyObject_CallFunctionObjArgs(python_function,
                                                        python_path_to_file,
                                                        NULL);
    
    // `load_via_pickle` is not responsible for decrefing `python_function`.
    Py_DECREF(python_path_to_file);
    if (!python_loaded_object)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        else
        {
            fprintf(stderr, "`PyObject_CallFunctionObjArgs` fails but the error indicator is not set.\n");
        }
        return NULL;
    }
    return python_loaded_object;
}


