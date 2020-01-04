#include "tests.h"

int test_append_sys_path(const std::string& path_to_additional_directory)
{
    /*
    All pointers to Python objects are
    declared below.
    */
    PyObject* python_path_sys(NULL);
    PyObject* python_item(NULL);
    
    if (append_sys_path(path_to_additional_directory) < 0)
    {
        return -1;
    }
    
    /*
    `PySys_GetObject` returns a borrowed reference.
    In Python 2.7, the argument of `PySys_GetObject`
    is `char*`. However, in Python 3.6, this argument
    is `const char*`.
    */
    std::string string_path("path");
    python_path_sys = PySys_GetObject(const_cast<char*>(string_path.c_str()));
    if (!python_path_sys)
    {
        fprintf(stderr, "`PySys_GetObject` fails.\n");
        return -1;
    }
    
    /*
    If the argument of `PyList_Size` is not a list
    object, a `SystemError` exception is thrown. The
    origin of the problem is not clearly described
    by this exception.
    */
    const bool error_bool(PyList_Check(python_path_sys));
    if (!error_bool)
    {
        fprintf(stderr, "`python_path_sys` is not a list object.\n");
        return -1;
    }
    Py_ssize_t python_length_list(PyList_Size(python_path_sys));
    std::string message("");
    char* ptr_message_portion(NULL);
    for (Py_ssize_t i(0); i < python_length_list; i++)
    {
        // `PyList_GetItem` returns a borrowed reference.
        python_item = PyList_GetItem(python_path_sys, i);
        if (!python_item)
        {
            if (PyErr_Occurred())
            {
                PyErr_Print();
            }
            else
            {
                fprintf(stderr, "`PyList_GetItem` fails but the error indicator is not set.\n");
            }
            return -1;
        }
        
        /*
        If `python_item` is not a string object at all,
        `PyString_AsString` returns NULL and raises `TypeError`.
        */
#if PY_MAJOR_VERSION <= 2
        ptr_message_portion = PyString_AsString(python_item);
#else
        ptr_message_portion = PyUnicode_AsUTF8(python_item);
#endif
        if (!ptr_message_portion)
        {
            if (PyErr_Occurred())
            {
                PyErr_Print();
            }
            else
            {
                fprintf(stderr, "`PyString_AsString` fails but the error indicator is not set.\n");
            }
            return -1;
        }
        const std::string message_portion(ptr_message_portion);
        message += message_portion + " ";
    }
    fprintf(stdout, "Python sys path:\n");
    fprintf(stdout, "%s\n", message.c_str());
    return 0;
}

int test_check_array_sorted_strictly_ascending()
{
    const unsigned int size_array(6);
    const float ptr_array[size_array] = {-44.12, -22.21, -8.78, 0.01, 3.02, 3.02};
    bool is_sorted_strictly_ascending(false);
    const int error_code(check_array_sorted_strictly_ascending<float>(ptr_array,
                                                                      size_array,
                                                                      is_sorted_strictly_ascending));
    if (error_code < 0)
    {
        return -1;
    }
    if (is_sorted_strictly_ascending)
    {
        fprintf(stdout, "The array is sorted in strictly ascending order.\n");
    }
    else
    {
        fprintf(stdout, "The array is not sorted in strictly ascending order.\n");
    }
    return 0;
}

int test_check_overlapping_intra_pattern_context_portions()
{
    const int width_target(8);
    const float mean_training(112.1);
    const int width_portion_above(3*width_target);
    
    /*
    `ptr_portion_above` is a pointer to the buffer
    of the masked context portion located above the
    target patch.
    */
    float ptr_portion_above[width_target*width_portion_above] = {0.};
    
    /*
    `ptr_portion_left` is a pointer to the buffer
    of the masked context portion located on the left
    side of the target patch.
    */
    float ptr_portion_left[2*width_target*width_target] = {0.};
    unsigned int i(0);
    
    /*
    The last row of the masked context portion located
    above the target patch is filled with non-zero values
    as a part of this row overlaps the 1st row of the
    intra pattern.
    */
    for (i = 0; i < width_portion_above; i++)
    {
        ptr_portion_above[(width_target - 1)*width_portion_above + i] = static_cast<float>(i) - mean_training;
    }
    
    /*
    The last column of the masked context portion located
    on the left side of the target patch is filled with non-zero
    values as a part of this column overlaps the 1st column
    of the intra pattern.
    */
    for (i = 0; i < 2*width_target; i++)
    {
        ptr_portion_left[(i + 1)*width_target - 1] = static_cast<float>(i) - mean_training;
    }
    
    /*
    `ptr_intra_pattern` is a pointer to the buffer
    of the intra pattern.
    */
    int ptr_intra_pattern[(2*width_target + 1)*(2*width_target + 1)] = {0};
    for (i = 0; i < width_target + 1; i++)
    {
        ptr_intra_pattern[i] = i + width_target - 1;
    }
    for (i = 0; i < width_target; i++)
    {
        ptr_intra_pattern[(i + 1)*(2*width_target + 1)] = i;
    }
    const int error_code(check_overlapping_intra_pattern_context_portions(ptr_portion_above,
                                                                          ptr_portion_left,
                                                                          ptr_intra_pattern,
                                                                          width_target,
                                                                          mean_training));
    if (error_code < 0)
    {
        return -1;
    }
    else
    {
        std::cout << "No error occurs during the checking of the overlapping between the intra pattern of the target patch and the two masked context portions." << std::endl;
        return 0;
    }
}

void test_create_tensors_context_portion()
{
    std::vector<tensorflow::Tensor> tensors_portion_above;
    std::vector<tensorflow::Tensor> tensors_portion_left;
    create_tensors_context_portion(tensors_portion_above,
                                   tensors_portion_left);
    std::vector<tensorflow::Tensor>::iterator it;
    
    // The initial value of `nb_dims` is meaningless.
    int nb_dims(0);
    const tensorflow::TensorShape* ptr_shape_tensor(NULL);
    std::cout << "Vector of three masked context portions located above the target patch." << std::endl;
    for (it = tensors_portion_above.begin(); it != tensors_portion_above.end(); it++)
    {
        nb_dims = it->dims();
        ptr_shape_tensor = &it->shape();
        std::cout << "Masked context portion located above the target patch of width " << pow(2, 4 + it - tensors_portion_above.begin()) << " pixels." << std::endl;
        for (int i(0); i < nb_dims; i++)
        {
            std::cout << "Dimension of index " << i << ": " << ptr_shape_tensor->dim_size(i) << std::endl;
        }
    }
    std::cout << "\nVector of three masked context portions located on the left side of the target patch." << std::endl;
    for (it = tensors_portion_left.begin(); it != tensors_portion_left.end(); it++)
    {
        nb_dims = it->dims();
        ptr_shape_tensor = &it->shape();
        std::cout << "Masked context portion located on the left side of the target patch of width " << pow(2, 4 + it - tensors_portion_left.begin()) << " pixels." << std::endl;
        for (int i(0); i < nb_dims; i++)
        {
            std::cout << "Dimension of index " << i << ": " << ptr_shape_tensor->dim_size(i) << std::endl;
        }
    }
}

void test_create_tensors_flattened_context()
{
    std::vector<tensorflow::Tensor> tensors_flattened_context;
    create_tensors_flattened_context(tensors_flattened_context);
    std::vector<tensorflow::Tensor>::iterator it;
    
    // The initial value of `nb_dims` is meaningless.
    int nb_dims(0);
    const tensorflow::TensorShape* ptr_shape_tensor(NULL);
    for (it = tensors_flattened_context.begin(); it != tensors_flattened_context.end(); it++)
    {
        nb_dims = it->dims();
        
        /*
        The precedence of ...&... (address of) is lower
        than the precedence of ...-> (member access).
        */
        ptr_shape_tensor = &it->shape();
        std::cout << "Flattened masked context for the target patch of width " << pow(2, 2 + it - tensors_flattened_context.begin()) << " pixels." << std::endl;
        for (int i(0); i < nb_dims; i++)
        {
            std::cout << "Dimension of index " << i << ": " << ptr_shape_tensor->dim_size(i) << std::endl;
        }
    }
}

int test_extract_context_portions_4_8(const int& uiTuWidthHeight)
{
    int error_code(0);
    
    /*
    The precedence of ...&&... (bitwise AND) is lower
    than the precedence of ...!=... (relation operator
    not equal).
    */
    if (uiTuWidthHeight != 4 && uiTuWidthHeight != 8)
    {
        std::cerr << "`uiTuWidthHeight` does not belong to {4, 8}." << std::endl;
        return -1;
    }
    const int heightDecodedChannel(32);
    const int iPicStride(40);
    
    /*
    It is easier to check the function `extract_context_portions`
    when `meanTraining` is equal to 0.
    */
    const float meanTraining(0.);
    
    // `decodedChannel` is the decoded channel buffer.
    int decodedChannel[heightDecodedChannel*iPicStride] = {0};
    for (int i(0); i < heightDecodedChannel*iPicStride; i++)
    {
        decodedChannel[i] = i;
    }
    
    /*
    A single buffer gathers the buffer of the masked context
    portion located above the current TB and the buffer of
    the masked context portion located on the left side of
    the current TB.
    */
    float piFlattenedContext[5*64*64] = {0.};
    float* const piPortionLeft(piFlattenedContext + 3*uiTuWidthHeight*uiTuWidthHeight);
    
    /*
    If the width of the current TB is equal to 64 pixels,
    there are 32 4x4 units on the left side of the current
    TB, 32 4x4 units above the current TB and one 4x4 unit
    above and on the left side of the current TB.
    */
    bool bNeighborFlags[65];
    for (int i(0); i < 65; i++)
    {
        bNeighborFlags[i] = true;
    }
    const int iAboveUnits(uiTuWidthHeight/2);
    const int iLeftUnits(uiTuWidthHeight/2);
    int iNumIntraNeighbor(iAboveUnits + iLeftUnits + 1);
    const int unitWidth(4);
    const int unitHeight(4);
    
    /*
    The value of the pixel located at the top-left
    corner of the current TB is equal to 412.
    */
    const int* const piRoiOrigin(decodedChannel + 10*iPicStride + 12);
    error_code = extract_context_portions(piRoiOrigin,
                                          piFlattenedContext,
                                          piPortionLeft,
                                          bNeighborFlags,
                                          iNumIntraNeighbor,
                                          unitWidth,
                                          unitHeight,
                                          iAboveUnits,
                                          iLeftUnits,
                                          uiTuWidthHeight,
                                          uiTuWidthHeight,
                                          iPicStride,
                                          meanTraining);
    if (error_code < 0)
    {
        return -1;
    }
    std::string message("");
    for (int i(0); i < 5*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piFlattenedContext[i]) != piFlattenedContext[i])
        {
            std::cerr << "An element of the flattened masked context buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piFlattenedContext[i])) + " ";
    }
    std::cout << "1st test:" << std::endl;
    std::cout << "Buffer of the flattened masked context filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    if (uiTuWidthHeight == 4)
    {
        std::cout << "248 -> 259 288 -> 299 328 -> 339 368 -> 379 408 -> 411 ... 688 -> 691" << std::endl;
    }
    else
    {
        std::cout << "84 -> 107 124 -> 147 164 -> 187 204 -> 227 244 -> 267 284 -> 307 324 -> 347 364 -> 387 404 -> 411 444 -> 451 484 -> 491 524 -> 531 ... 1004 -> 1011" << std::endl;
    }
    
    /*
    In the 1st test, all the neighbouring 4x4 units were available.
    However, in the 2nd test, on the left side of the current TB,
    the bottommost neighbouring 4x4 unit is not available.
    */
    bNeighborFlags[0] = false;
    iNumIntraNeighbor--;
    error_code = extract_context_portions(piRoiOrigin,
                                          piFlattenedContext,
                                          piPortionLeft,
                                          bNeighborFlags,
                                          iNumIntraNeighbor,
                                          unitWidth,
                                          unitHeight,
                                          iAboveUnits,
                                          iLeftUnits,
                                          uiTuWidthHeight,
                                          uiTuWidthHeight,
                                          iPicStride,
                                          meanTraining);
    if (error_code < 0)
    {
        return -1;
    }
    message = "";
    for (int i(0); i < 5*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piFlattenedContext[i]) != piFlattenedContext[i])
        {
            std::cerr << "An element of the flattened masked context buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piFlattenedContext[i])) + " ";
    }
    std::cout << "\n2nd test:" << std::endl;
    std::cout << "Buffer of the flattened masked context filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    if (uiTuWidthHeight == 4)
    {
        std::cout << "248 -> 259 288 -> 299 328 -> 339 368 -> 379 408 -> 411 448 -> 451 488 -> 491 528 -> 531 {16 times zero}." << std::endl;
    }
    else
    {
        std::cout << "84 -> 107 124 -> 147 164 -> 187 204 -> 227 244 -> 267 284 -> 307 324 -> 347 364 -> 387 404 -> 411 ... 844 -> 851 {32 times zero}." << std::endl;
    }
    
    /*
    In the 2nd test, all the neighbouring 4x4 units were available.
    However, in the 3rd test, above the current TB, the rightmost
    neighbouring 4x4 unit is not available.
    */
    bNeighborFlags[0] = true;
    bNeighborFlags[iLeftUnits + iAboveUnits] = false;
    error_code = extract_context_portions(piRoiOrigin,
                                          piFlattenedContext,
                                          piPortionLeft,
                                          bNeighborFlags,
                                          iNumIntraNeighbor,
                                          unitWidth,
                                          unitHeight,
                                          iAboveUnits,
                                          iLeftUnits,
                                          uiTuWidthHeight,
                                          uiTuWidthHeight,
                                          iPicStride,
                                          meanTraining);
    if (error_code < 0)
    {
        return -1;
    }
    message = "";
    for (int i(0); i < 5*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piFlattenedContext[i]) != piFlattenedContext[i])
        {
            std::cerr << "An element of the flattened masked context buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piFlattenedContext[i])) + " ";
    }
    std::cout << "\n3rd test:" << std::endl;
    std::cout << "Buffer of the flattened masked context filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    if (uiTuWidthHeight == 4)
    {
        std::cout << "248 -> 255 0 0 0 0 288 -> 295 0 0 0 0 328 -> 335 0 0 0 0 368 -> 375 0 0 0 0 408 -> 411 ... 688 -> 691" << std::endl;
    }
    else
    {
        std::cout << "84 -> 103 {4 times zero} 124 -> 143 {4 times zero} 164 -> 183 { 4 times zero} 204 -> 223 {4 times zero} 244 -> 263 {4 times zero} 284 -> 303 {4 times zero} 324 -> 343 {4 times zero} 364 -> 383 {4 times zero} 404 -> 411 ... 1004 -> 1011" << std::endl;
    }
    return 0;
}

int test_extract_context_portions_16()
{
    int error_code(0);
    const int uiTuWidthHeight(16);
    const int heightDecodedChannel(56);
    const int iPicStride(60);
    
    /*
    It is easier to check the function `extract_context_portions`
    when `meanTraining` is equal to 0.
    */
    const float meanTraining(0.);
    int decodedChannel[heightDecodedChannel*iPicStride] = {0};
    for (int i(0); i < heightDecodedChannel*iPicStride; i++)
    {
        decodedChannel[i] = i;
    }
    float piPortionAbove[3*64*64] = {0.};
    float piPortionLeft[2*64*64] = {0.};
    
    /*
    If the width of the current TB is equal to 64 pixels,
    there are 32 4x4 units on the left side of the current
    TB, 32 4x4 units above the current TB and one 4x4 unit
    on the left side/above the current TB.
    */
    bool bNeighborFlags[65];
    for (int i(0); i < 65; i++)
    {
        bNeighborFlags[i] = true;
    }
    const int iAboveUnits(uiTuWidthHeight/2);
    const int iLeftUnits(uiTuWidthHeight/2);
    int iNumIntraNeighbor(iAboveUnits + iLeftUnits + 1);
    const int unitWidth(4);
    const int unitHeight(4);
    
    /*
    The value of the pixel located at the top-left
    corner of the current TB is equal to 1218.
    */
    const int* const piRoiOrigin(decodedChannel + 20*iPicStride + 18);
    error_code = extract_context_portions(piRoiOrigin,
                                          piPortionAbove,
                                          piPortionLeft,
                                          bNeighborFlags,
                                          iNumIntraNeighbor,
                                          unitWidth,
                                          unitHeight,
                                          iAboveUnits,
                                          iLeftUnits,
                                          uiTuWidthHeight,
                                          uiTuWidthHeight,
                                          iPicStride,
                                          meanTraining);
    if (error_code < 0)
    {
        return -1;
    }
    std::string message("");
    for (int i(0); i < 3*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piPortionAbove[i]) != piPortionAbove[i])
        {
            std::cerr << "An element of the masked above portion buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piPortionAbove[i])) + " ";
    }
    std::cout << "1st test:" << std::endl;
    std::cout << "Buffer of the masked context portion located above the current TB filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    std::cout << "242 -> 289 302 -> 349 362 -> 409 422 -> 469 ... 1142 -> 1189" << std::endl;
    message = "";
    for (int i(0); i < 2*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piPortionLeft[i]) != piPortionLeft[i])
        {
            std::cerr << "An element of the masked left portion buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piPortionLeft[i])) + " ";
    }
    std::cout << "Buffer of the masked context portion located on the left side of the current TB filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    std::cout << "1202 -> 1217 1262 -> 1277 1322 -> 1337 1382 -> 1397 ... 3062 -> 3077" << std::endl;
    
    /*
    In the 1st test, all the neighbouring 4x4 units were available.
    However, in the 2nd test, on the left side of the current TB,
    the two bottommost neighbouring 4x4 units are not available.
    */
    bNeighborFlags[0] = false;
    bNeighborFlags[1] = false;
    iNumIntraNeighbor -= 2;
    error_code = extract_context_portions(piRoiOrigin,
                                          piPortionAbove,
                                          piPortionLeft,
                                          bNeighborFlags,
                                          iNumIntraNeighbor,
                                          unitWidth,
                                          unitHeight,
                                          iAboveUnits,
                                          iLeftUnits,
                                          uiTuWidthHeight,
                                          uiTuWidthHeight,
                                          iPicStride,
                                          meanTraining);
    if (error_code < 0)
    {
        return -1;
    }
    message = "";
    for (int i(0); i < 3*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piPortionAbove[i]) != piPortionAbove[i])
        {
            std::cerr << "An element of the masked above portion buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piPortionAbove[i])) + " ";
    }
    std::cout << "\n2nd test:" << std::endl;
    std::cout << "Buffer of the masked context portion located above the current TB filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    std::cout << "242 -> 289 302 -> 349 362 -> 409 422 -> 469 ... 1142 -> 1189" << std::endl;
    message = "";
    for (int i(0); i < 2*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piPortionLeft[i]) != piPortionLeft[i])
        {
            std::cerr << "An element of the masked left portion buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piPortionLeft[i])) + " ";
    }
    std::cout << "Buffer of the masked context portion located on the left side of the current TB filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    std::cout << "1202 -> 1217 1262 -> 1277 1322 -> 1337 1382 -> 1397 ... 2582 -> 2597 {128 times zero}." << std::endl;
    
    /*
    In the 1st test, all the neighbouring 4x4 units were available.
    However, in the 3rd test, above the current TB, the rightmost
    neighbouring 4x4 unit is not available.
    */
    bNeighborFlags[0] = true;
    bNeighborFlags[1] = true;
    bNeighborFlags[iLeftUnits + iAboveUnits] = false;
    iNumIntraNeighbor++;
    error_code = extract_context_portions(piRoiOrigin,
                                          piPortionAbove,
                                          piPortionLeft,
                                          bNeighborFlags,
                                          iNumIntraNeighbor,
                                          unitWidth,
                                          unitHeight,
                                          iAboveUnits,
                                          iLeftUnits,
                                          uiTuWidthHeight,
                                          uiTuWidthHeight,
                                          iPicStride,
                                          meanTraining);
    if (error_code < 0)
    {
        return -1;
    }
    message = "";
    for (int i(0); i < 3*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piPortionAbove[i]) != piPortionAbove[i])
        {
            std::cerr << "An element of the masked above portion buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piPortionAbove[i])) + " ";
    }
    std::cout << "\n3rd test:" << std::endl;
    std::cout << "Buffer of the masked context portion located above the current TB filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer:" << std::endl;
    std::cout << "242 -> 285 0 0 0 0 302 -> 345 0 0 0 0 362 -> 405 0 0 0 0 422 -> 465 0 0 0 0 ... 1142 -> 1185 0 0 0 0" << std::endl;
    message = "";
    for (int i(0); i < 2*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        if (std::floor(piPortionLeft[i]) != piPortionLeft[i])
        {
            std::cerr << "An element of the masked left portion buffer is not a whole number." << std::endl;
            return -1;
        }
        message += std::to_string(static_cast<int>(piPortionLeft[i])) + " ";
    }
    std::cout << "Buffer of the masked context portion located on the left side of the current TB filled by the function:" << std::endl;
    std::cout << message << std::endl;
    std::cout << "Expected buffer: same as in the 1st test." << std::endl;
    return 0;
}

int test_fill_map_intra_prediction_modes()
{
    const unsigned int height_map(96);
    const unsigned int width_map(128);
    unsigned char ptr_map[height_map*width_map] = {0};
    unsigned int i(0);
    for (i = 0; i < height_map*width_map; i++)
    {
        ptr_map[i] = 255;
    }
    const unsigned int height_ctu_units(16);
    const unsigned int width_ctu_units(16);
    
    /*
    The pixel at the top-left of the current CTB in the
    map of intra prediction modes is located at the position
    [16, 8] in this map.
    */
    const unsigned int row_ctu_in_map(16);
    const unsigned int column_ctu_in_map(8);
    unsigned char* const ptr_ctu_map(ptr_map + row_ctu_in_map*width_map + column_ctu_in_map);
    unsigned int j(0);
    for (i = 0; i < 4*height_ctu_units; i++)
    {
        for (j = 0; j < 4*width_ctu_units; j++)
        {
            ptr_ctu_map[i*width_map + j] = 200;
        }
    }
    const unsigned int address_cu_in_ctu_units(52);
    
    /*
    On a drawing of the Z-scanning of the CBs in a CTB, we see
    that, when `address_cu_in_ctu_units` is equal to 52, the
    pixel at the top-left of the current CB is located at the
    position [24, 8] in the current CTB.
    */
    const unsigned int row_cu_in_map(row_ctu_in_map + 24);
    const unsigned int column_cu_in_map(column_ctu_in_map + 8);
    
    /*
    `ptr_intra_prediction_modes` is a pointer to the 1st element
    in the array storing the intra prediction mode of each PB
    in the current CTB.
    */
    unsigned char ptr_intra_prediction_modes[height_ctu_units*width_ctu_units] = {0};
    for (i = 0; i < 16; i++)
    {
        ptr_intra_prediction_modes[address_cu_in_ctu_units + i] = 32*(i/4);
    }
    
    /*
    The height and the width of the current CB are equal to
    16 pixels.
    */
    unsigned char ptr_height_cus[height_ctu_units*width_ctu_units] = {0};
    ptr_height_cus[address_cu_in_ctu_units] = 16;
    unsigned char ptr_width_cus[height_ctu_units*width_ctu_units] = {0};
    ptr_width_cus[address_cu_in_ctu_units] = 16;
    const unsigned int nb_pus_in_cu(4);
    
    // As there are 4 PBs in the CB, each PB contains 4 units.
    const unsigned int offset_pu_in_cu_units(4);
    int error_code(0);
    error_code = fill_map_intra_prediction_modes(ptr_map,
                                                 height_map,
                                                 width_map,
                                                 ptr_intra_prediction_modes,
                                                 ptr_height_cus,
                                                 ptr_width_cus,
                                                 row_cu_in_map,
                                                 column_cu_in_map,
                                                 address_cu_in_ctu_units,
                                                 nb_pus_in_cu,
                                                 offset_pu_in_cu_units);
    if (error_code < 0)
    {
        return -1;
    }
    const std::string path("hevc/hm_common/c++/pseudo_visualization/fill_map_intra_prediction_modes.pgm");
    error_code = visualize_channel<unsigned char>(ptr_map,
                                                  height_map,
                                                  width_map,
                                                  255,
                                                  path);
    if (error_code < 0)
    {
        return -1;
    }
    return 0;
}

void test_find_position_in_array()
{
    const unsigned int size_array(9);
    const float ptr_array[size_array] = {0.1, 128.3, -127.2, 7., 4., -3., -5., 4.4, 0.};
    const float value(-127.2);
    const int found_position(find_position_in_array<float>(ptr_array,
                                                           size_array,
                                                           value));
    std::cout << "Position of the given value in the array: " << found_position << std::endl;
}

int test_get_callable(const std::string& path_to_additional_directory)
{
    /*
    All pointers to Python objects are
    declared below.
    */
    PyObject* python_function(NULL);
    
    if (append_sys_path(path_to_additional_directory) < 0)
    {
        return -1;
    }
    
    /*
    After calling the function `get_callable`,
    we are responsible for decrefing `python_function`.
    */
    python_function = get_callable("loading", "load_via_pickle");
    if (!python_function)
    {
        return -1;
    }
    Py_DECREF(python_function);
    fprintf(stdout, "No error occurs while getting the function `load_via_pickle` in the file \"loading.py\".\n");
    return 0;
}

void test_is_string_special_characters_exclusively()
{
    const std::string string_in("    \n \t  \f");
    if (is_string_special_characters_exclusively(string_in))
    {
        std::cout << "The given string only contains special characters." << std::endl;
    }
    else
    {
        std::cout << "The given string contains at least one non-special character." << std::endl;
    }
}

int test_load_graph()
{
    std::unique_ptr<tensorflow::Session> unique_ptr_session;
    tensorflow::Status status_load_graph(load_graph("hevc/hm_common/c++/pseudo_data/width_target_4/graph_output.pbtxt", unique_ptr_session));
    if (!status_load_graph.ok())
    {
        LOG(ERROR) << status_load_graph;
        return -1;
    }
    return 0;
}

int test_load_graphs()
{
    std::vector<std::unique_ptr<tensorflow::Session>> vector_unique_ptrs_session;
    vector_unique_ptrs_session.push_back(std::unique_ptr<tensorflow::Session>());
    vector_unique_ptrs_session.push_back(std::unique_ptr<tensorflow::Session>());
    std::vector<std::string> vector_paths_to_graphs_output;
    vector_paths_to_graphs_output.push_back(
        std::string("hevc/hm_common/c++/pseudo_data/width_target_4/graph_output.pbtxt")
    );
    vector_paths_to_graphs_output.push_back(
        std::string("hevc/hm_common/c++/pseudo_data/width_target_8/graph_output.pbtxt")
    );
    tensorflow::Status status_load_graph(load_graphs(vector_paths_to_graphs_output,
                                                     vector_unique_ptrs_session));
    if (!status_load_graph.ok())
    {
        LOG(ERROR) << status_load_graph;
        return -1;
    }
    return 0;
}

int test_load_via_pickle(const std::string& path_to_additional_directory)
{
    /*
    All pointers to Python objects are
    declared below.
    */
    PyObject* python_function(NULL);
    PyObject* python_loaded_object(NULL);
    
    if (append_sys_path(path_to_additional_directory) < 0)
    {
        return -1;
    }
    
    /*
    After calling the function `get_callable`,
    we are responsible for decrefing `python_function`.
    */
    python_function = get_callable("loading", "load_via_pickle");
    if (!python_function)
    {
        return -1;
    }
    const std::string path_to_file("hevc/hm_common/c++/pseudo_data/pseudo_integer.pkl");
    
    /*
    After calling the function `load_via_pickle`, we
    are responsible for decrefing `python_loaded_object`.
    */
    python_loaded_object = load_via_pickle(python_function, path_to_file);
    Py_DECREF(python_function);
    if (!python_loaded_object)
    {
        return -1;
    }
    bool error_bool(false);
#if PY_MAJOR_VERSION <= 2
    error_bool = PyInt_CheckExact(python_loaded_object);
#else
    error_bool = PyLong_CheckExact(python_loaded_object);
#endif
    if (!error_bool)
    {
        fprintf(stderr, "`python_loaded_object` does not points to a Python object of type `PyInt_Type`.\n");
        Py_DECREF(python_loaded_object);
        return -1;
    }
    long output_long(0);
#if PY_MAJOR_VERSION <= 2
    output_long = PyInt_AsLong(python_loaded_object);
#else
    output_long = PyLong_AsLong(python_loaded_object);
#endif
    Py_DECREF(python_loaded_object);
    
    /*
    An error occurs if `PyInt_AsLong` returns -1 and
    `PyErr_Occurred` returns true, see <https://docs.python.org/2/c-api/int.html>.
    */
    if (output_long == -1)
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
            return -1;
        }
    }
    fprintf(stdout, "Loaded integer: %ld\n", output_long);
    return 0;
}

int test_parse_file_strings_one_key()
{
    std::map<unsigned int, std::string> map_storage;
    const std::string path_to_file("hevc/hm_common/c++/pseudo_data/pseudo_file_strings_one_key.txt");
    const std::string delimiters(",;:");
    const int error_code(parse_file_strings_one_key(map_storage,
                                                    path_to_file,
                                                    delimiters));
    if (error_code < 0)
    {
        return -1;
    }
    std::map<unsigned int, std::string>::const_iterator it;
    std::cout << "Map for storing each unsigned integer key and its path:" << std::endl;
    for (it = map_storage.begin(); it != map_storage.end(); it++)
    {
        std::cout << "Key: " << it->first << ", value: " << it->second << std::endl;
    }
    return 0;
}

int test_parse_file_strings_three_keys()
{
    std::map<std::pair<unsigned int, unsigned int>, std::string> map_false;
    std::map<std::pair<unsigned int, unsigned int>, std::string> map_true;
    const std::string path_to_file("hevc/hm_common/c++/pseudo_data/pseudo_file_strings_three_keys.txt");
    const std::string delimiters(",;");
    const int error_code(parse_file_strings_three_keys(map_false,
                                                       map_true,
                                                       path_to_file,
                                                       delimiters));
    if (error_code < 0)
    {
        return -1;
    }
    std::map<std::pair<unsigned int, unsigned int>, std::string>::const_iterator it;
    std::cout << "Map of strings with false as boolean key:" << std::endl;
    for (it = map_false.begin(); it != map_false.end(); it++)
    {
        std::cout << "Key: {" << (it->first).first << ", " << (it->first).second << "}, value: " << it->second << std::endl;
    }
    std::cout << "Map of strings with true as boolean key:" << std::endl;
    for (it = map_true.begin(); it != map_true.end(); it++)
    {
        std::cout << "Key: {" << (it->first).first << ", " << (it->first).second << "}, value: " << it->second << std::endl;
    }
    return 0;
}

int test_prediction_neural_network_convolutional()
{
    // The height and the width of the target patch are equal.
    const int uiTuWidthHeight(16);
    tensorflow::Tensor tensor_portion_above(tensorflow::DT_FLOAT,
                                            {1, uiTuWidthHeight, 3*uiTuWidthHeight, 1});
    tensorflow::Tensor tensor_portion_left(tensorflow::DT_FLOAT,
                                           {1, 2*uiTuWidthHeight, uiTuWidthHeight, 1});
    float* const ptr_portion_above(tensor_portion_above.flat<float>().data());
    float* const ptr_portion_left(tensor_portion_left.flat<float>().data());
    
    /*
    Let's say that the mean pixels intensity over different
    luminance images is already subtracted from the values
    below.
    */
    for (int i(0); i < 3*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        ptr_portion_above[i] = -80.;
    }
    for (int i(0); i < 2*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        ptr_portion_left[i] = -80.;
    }
    for (int i(0); i < uiTuWidthHeight; i++)
    {
        ptr_portion_above[uiTuWidthHeight + 1 + i*3*uiTuWidthHeight] = 20.;
    }
    
    // The graph and the parameters are loaded.
    std::unique_ptr<tensorflow::Session> unique_ptr_session;
    tensorflow::Status status_load_graph(load_graph("hevc/hm_common/c++/pseudo_data/width_target_16/graph_output.pbtxt", unique_ptr_session));
    if (!status_load_graph.ok())
    {
        LOG(ERROR) << status_load_graph;
        return -1;
    }
    
    /*
    The prediction of the target patch is computed from the two
    masked context portions.
    */
    std::vector<tensorflow::Tensor> vector_tensor_prediction;
    const tensorflow::string name_input_above("node_portion_above");
    const tensorflow::string name_input_left("node_portion_left");
    const tensorflow::string name_output("convolutional/merger/transpose_convolution_3/node_output");
    tensorflow::Status status_run(
        unique_ptr_session->Run(
            {
                {name_input_above, tensor_portion_above},
                {name_input_left, tensor_portion_left}
            },
            {name_output},
            {},
            &vector_tensor_prediction
        )
    );
    if (!status_run.ok())
    {
        LOG(ERROR) << status_run;
        return -1;
    }
    const tensorflow::Tensor& tensor_prediction(vector_tensor_prediction.at(0));
    
    /*
    The number of dimensions of `tensor_prediction` has to be
    equal to 4.
    */
    const int nb_dims(tensor_prediction.dims());
    if (nb_dims != 4)
    {
        fprintf(stderr, "`nb_dims` is not equal to 4.\n");
        return -1;
    }
    const tensorflow::TensorShape* ptr_shape_tensor(&tensor_prediction.shape());
    
    /*
    The shape of `tensor_prediction` has to be
    (1, `uiTuWidthHeight`, `uiTuWidthHeight`, 1).
    */
    if (ptr_shape_tensor->dim_size(0) != 1 || ptr_shape_tensor->dim_size(3) != 1)
    {
        fprintf(stderr, "Either the 1st dimension or the 4th dimension of the prediction tensor is not equal to 1.\n");
        return -1;
    }
    if (ptr_shape_tensor->dim_size(1) != uiTuWidthHeight || ptr_shape_tensor->dim_size(2) != uiTuWidthHeight)
    {
        fprintf(stderr, "Either the 2nd dimension or the 3rd dimension of the prediction tensor is not equal to %d.\n", uiTuWidthHeight);
        return -1;
    }
    
    /*
    The 2nd column of the output of the convolutional prediction
    neural network must contain values relatively close to 20.0.
    */
    const float* const ptr_prediction(tensor_prediction.flat<float>().data());
    std::string message("");
    for (int i(0); i < uiTuWidthHeight; i++)
    {
        for (int j(0); j < uiTuWidthHeight; j++)
        {
            message += std::to_string(ptr_prediction[i*uiTuWidthHeight + j]) + " ";
        }
        message += "\n";
    }
    fprintf(stdout, "Output of the convolutional prediction neural network:\n");
    fprintf(stdout, "%s\n", message.c_str());
    return 0;
}

int test_prediction_neural_network_fully_connected()
{
    // The height and the width of the target patch are equal.
    const int uiTuWidthHeight(4);
    tensorflow::Tensor tensor_flattened_context(tensorflow::DT_FLOAT,
                                                {1, 5*uiTuWidthHeight*uiTuWidthHeight});
    float* const ptr_flattened_context(tensor_flattened_context.flat<float>().data());
    
    /*
    Let's say that the mean pixels intensity over different
    luminance images is already subtracted from the values
    below.
    */
    for (int i(0); i < 5*uiTuWidthHeight*uiTuWidthHeight; i++)
    {
        ptr_flattened_context[i] = -100.;
    }
    
    /*
    A vertical line is inside the context portion
    located above the target patch.
    */
    for (int i(0); i < uiTuWidthHeight; i++)
    {
        ptr_flattened_context[uiTuWidthHeight + 1 + i*3*uiTuWidthHeight] = 0.;
    }
    
    // The graph and the parameters are loaded.
    std::unique_ptr<tensorflow::Session> unique_ptr_session;
    tensorflow::Status status_load_graph(load_graph("hevc/hm_common/c++/pseudo_data/width_target_4/graph_output.pbtxt", unique_ptr_session));
    if (!status_load_graph.ok())
    {
        LOG(ERROR) << status_load_graph;
        return -1;
    }
    
    // The prediction of the target patch is computed from the flattened masked context.
    const tensorflow::string name_input("node_flattened_context");
    const tensorflow::string name_output("fully_connected/node_output");
    std::vector<tensorflow::Tensor> vector_tensor_prediction;
    tensorflow::Status status_run(
        unique_ptr_session->Run(
            {
                {name_input, tensor_flattened_context}
            },
            {name_output},
            {},
            &vector_tensor_prediction
        )
    );
    if (!status_run.ok())
    {
        LOG(ERROR) << status_run;
        return -1;
    }
    const tensorflow::Tensor& tensor_prediction(vector_tensor_prediction.at(0));
    
    /*
    The number of dimensions of `tensor_prediction` has to be
    equal to 4.
    */
    const int nb_dims(tensor_prediction.dims());
    if (nb_dims != 4)
    {
        fprintf(stderr, "`nb_dims` is not equal to 4.\n");
        return -1;
    }
    const tensorflow::TensorShape* ptr_shape_tensor(&tensor_prediction.shape());
    
    /*
    The shape of `tensor_prediction` has to be
    (1, `uiTuWidthHeight`, `uiTuWidthHeight`, 1).
    */
    if (ptr_shape_tensor->dim_size(0) != 1 || ptr_shape_tensor->dim_size(3) != 1)
    {
        fprintf(stderr, "Either the 1st dimension or the 4th dimension of the prediction tensor is not equal to 1.\n");
        return -1;
    }
    if (ptr_shape_tensor->dim_size(1) != uiTuWidthHeight || ptr_shape_tensor->dim_size(2) != uiTuWidthHeight)
    {
        fprintf(stderr, "Either the 2nd dimension or the 3rd dimension of the prediction tensor is not equal to %d.\n", uiTuWidthHeight);
        return -1;
    }
    
    /*
    The 1st, the 3rd and the 4th columns of the output of the
    fully-connected prediction neural network must contain values
    close to -100.0. The 2nd column of this output must contain
    values close to 0.0.
    */
    const float* const ptr_prediction(tensor_prediction.flat<float>().data());
    std::string message("");
    for (int i(0); i < uiTuWidthHeight; i++)
    {
        for (int j(0); j < uiTuWidthHeight; j++)
        {
            message += std::to_string(ptr_prediction[i*uiTuWidthHeight + j]) + " ";
        }
        message += "\n";
    }
    fprintf(stdout, "Output of the fully-connected prediction neural network:\n");
    fprintf(stdout, "%s\n", message.c_str());
    return 0;
}

void test_remove_leading_trailing_whitespaces()
{
    std::string string_0("");
    remove_leading_trailing_whitespaces(string_0);
    if (string_0.empty())
    {
        std::cout << "The 1st string after removing its leading and trailing whitespaces is empty." << std::endl;
    }
    else
    {
        std::cout << "The 1st string after removing its leading and trailing whitespaces is not empty." << std::endl;
    }
    std::string string_1("          Rajon Rondo");
    remove_leading_trailing_whitespaces(string_1);
    std::cout << "2nd string after removing its leading and trailing whitespaces: " << string_1 << std::endl;
    std::string string_2(" \t \v\t ");
    remove_leading_trailing_whitespaces(string_2);
    if (string_2.empty())
    {
        std::cout << "The 3rd string after removing its leading and trailing whitespaces is empty." << std::endl;
    }
    else
    {
        std::cout << "The 3rd string after removing its leading and trailing whitespaces is not empty." << std::endl;
    }
    std::string string_3("Demarcus Cousins      \t    ");
    remove_leading_trailing_whitespaces(string_3);
    std::cout << "4th string after removing its leading and trailing whitespaces: " << string_3 << std::endl;
}

int test_replace_in_array_by_value()
{
    const unsigned int size_arrays(6);
    int ptr_array_replacement[size_arrays] = {0, 6, -11, 2, -11, 2};
    int ptr_array_reference[size_arrays] = {1, 3, 444, 1, -77, 0};
    const int value_replacement(-11);
    std::string message_0("");
    std::string message_1("");
    unsigned int i(0);
    std::cout << "Value for the replacement: " << value_replacement << std::endl;
    for (i = 0; i < size_arrays; i++)
    {
        message_0 += std::to_string(ptr_array_replacement[i]) + " ";
        message_1 += std::to_string(ptr_array_reference[i]) + " ";
    }
    std::cout << "1st array before the replacement: " << message_0 << std::endl;
    std::cout << "2nd array before the replacement: " << message_1 << std::endl;
    const int error_code(replace_in_array_by_value<int>(ptr_array_replacement,
                                                        ptr_array_reference,
                                                        size_arrays,
                                                        value_replacement));
    if (error_code < 0)
    {
        return -1;
    }
    message_0 = "";
    for (i = 0; i < size_arrays; i++)
    {
        message_0 += std::to_string(ptr_array_replacement[i]) + " ";
    }
    std::cout << "1st array after the replacement: " << message_0 << std::endl;
    return 0;
}

void test_split_string()
{
    std::vector<std::string> vector_substrings;
    const std::string input_string("path_0;path_1,path_2:path_3-path_4;path_5");
    const std::string delimiters(":,-;");
    split_string(vector_substrings,
                 input_string,
                 delimiters);
    std::cout << "Input string: " << input_string << std::endl;
    std::cout << "Set of delimiters for splitting: " << delimiters << std::endl;
    std::cout << "Content of the vector of substrings:" << std::endl;
    std::vector<std::string>::const_iterator it;
    for (it = vector_substrings.begin(); it != vector_substrings.end(); it++)
    {
        std::cout << *it << std::endl;
    }
}

int test_visualize_channel()
{
    const std::string path("hevc/hm_common/c++/pseudo_visualization/visualize_channel.pgm");
    const unsigned int height_channel(32);
    const unsigned int width_channel(64);
    unsigned int ptr_channel[height_channel*width_channel] = {0};
    unsigned int i(0);
    for (i = 0; i < height_channel*width_channel; i++)
    {
        ptr_channel[i] = 200;
    }
    for (i = 0; i < height_channel; i++)
    {
        ptr_channel[i*width_channel + 2] = 80;
    }
    const int error_code(visualize_channel<unsigned int>(ptr_channel,
                                                         height_channel,
                                                         width_channel,
                                                         255,
                                                         path));
    if (error_code < 0)
    {
        return -1;
    }
    return 0;
}

int test_visualize_context_portions()
{
    const std::string path("hevc/hm_common/c++/pseudo_visualization/visualize_context_portions.pgm");
    const unsigned int width_target(64);
    
    // The mean pixels luminance is fake.
    const float mean_training(114.221);
    float ptr_portion_above[3*width_target*width_target] = {0.};
    float ptr_portion_left[2*width_target*width_target] = {0.};
    unsigned int i(0);
    for (i = 0; i < 3*width_target*width_target; i++)
    {
        ptr_portion_above[i] = 190. - mean_training;
    }
    for (i = 0; i < 2*width_target*width_target; i++)
    {
        ptr_portion_left[i] = 50. - mean_training;
    }
    const int error_code(visualize_context_portions(ptr_portion_above,
                                                    ptr_portion_left,
                                                    width_target,
                                                    mean_training,
                                                    path));
    if (error_code < 0)
    {
        return -1;
    }
    return 0;
}

int test_visualize_thresholded_channel()
{
    const unsigned int height_channel(256);
    const unsigned int width_channel(192);
    unsigned char ptr_channel[height_channel*width_channel] = {0};
    unsigned int i(0);
    unsigned int j(0);
    for (i = 99; i < height_channel; i++)
    {
        for (j = 99; j < width_channel; j++)
        {
            ptr_channel[i*width_channel + j] = 32;
        }
    }
    for (i = 4; i < 52; i++)
    {
        for (j = 24; j < 72; j++)
        {
            ptr_channel[i*width_channel + j] = 21;
        }
    }
    for (i = 0; i < 16; i++)
    {
        for (j = 0; j < 16; j++)
        {
            ptr_channel[i*width_channel + j] = 2;
        }
    }
    
    /*
    Three different thresholds and four different colors
    are defined. The colors are red, orange, blue and
    green.
    */
    const unsigned int size_arrays_thresholds(3);
    const unsigned char ptr_array_thresholds[size_arrays_thresholds] = {1, 20, 31};
    const unsigned int ptr_array_colors[size_arrays_thresholds + 1][3] = {
        {255, 0, 0},
        {255, 165, 0},
        {0, 0, 255},
        {0, 255, 0}
    };
    const std::string path("hevc/hm_common/c++/pseudo_visualization/visualize_thresholded_channel.ppm");
    const int error_code(visualize_thresholded_channel<unsigned char>(ptr_channel,
                                                                      height_channel,
                                                                      width_channel,
                                                                      ptr_array_thresholds,
                                                                      size_arrays_thresholds,
                                                                      ptr_array_colors,
                                                                      255,
                                                                      path));
    if (error_code < 0)
    {
        return -1;
    }
    return 0;
}


