#include "tests.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "At least one argument is required. The required argument is the name of the function to be tested." << std::endl;
        return -1;
    }
    
    /*
    The precedence of !... (logical NOT) is higher than
    the precedence of ...||... (logical OR).
    */
    if (!strcmp(argv[1], "append_sys_path") || !strcmp(argv[1], "get_callable") || !strcmp(argv[1], "load_via_pickle"))
    {
        if (argc != 3)
        {
            std::cerr << "To test the function named " << argv[1] << ", exactly two arguments are required." << std::endl;
            return -1;
        }
    }
    else
    {
        if (argc != 2)
        {
            std::cerr << "To test the function named " << argv[1] << ", exactly one argument is required." << std::endl;
            return -1;
        }
    }
    
    int error_code(0);
    if (!strcmp(argv[1], "append_sys_path"))
    {
        // The Python interpreter is initialized.
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            std::cerr << "The Python interpreter is not initialized." << std::endl;
            return -1;
        }
        error_code = test_append_sys_path(argv[2]);
        
        // The Python interpreter is shut down.
        Py_Finalize();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "check_array_sorted_strictly_ascending"))
    {
        error_code = test_check_array_sorted_strictly_ascending();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "check_overlapping_intra_pattern_context_portions"))
    {
        error_code = test_check_overlapping_intra_pattern_context_portions();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "create_tensors_context_portion"))
    {
        test_create_tensors_context_portion();
    }
    else if (!strcmp(argv[1], "create_tensors_flattened_context"))
    {
        test_create_tensors_flattened_context();
    }
    else if (!strcmp(argv[1], "extract_context_portions"))
    {
        std::cout << "\nTU width equals to 4 pixels." << std::endl;
        error_code = test_extract_context_portions_4_8(4);
        if (error_code < 0)
        {
            return -1;
        }
        std::cout << "\nTU width equals to 8 pixels." << std::endl;
        error_code = test_extract_context_portions_4_8(8);
        if (error_code < 0)
        {
            return -1;
        }
        std::cout << "\nTU width equals to 16 pixels." << std::endl;
        error_code = test_extract_context_portions_16();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "fill_map_intra_prediction_modes"))
    {
        error_code = test_fill_map_intra_prediction_modes();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "find_position_in_array"))
    {
        test_find_position_in_array();
    }
    else if (!strcmp(argv[1], "get_callable"))
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            std::cerr << "The Python interpreter is not initialized." << std::endl;
            return -1;
        }
        error_code = test_get_callable(argv[2]);
        Py_Finalize();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "is_string_special_characters_exclusively"))
    {
        test_is_string_special_characters_exclusively();
    }
    else if (!strcmp(argv[1], "load_graph"))
    {
        error_code = test_load_graph();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "load_graphs"))
    {
        error_code = test_load_graphs();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "load_via_pickle"))
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            std::cerr << "The Python interpreter is not initialized." << std::endl;
            return -1;
        }
        error_code = test_load_via_pickle(argv[2]);
        Py_Finalize();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "parse_file_strings_one_key"))
    {
        error_code = test_parse_file_strings_one_key();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "parse_file_strings_three_keys"))
    {
        error_code = test_parse_file_strings_three_keys();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "prediction_neural_network_convolutional"))
    {
        error_code = test_prediction_neural_network_convolutional();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "prediction_neural_network_fully_connected"))
    {
        error_code = test_prediction_neural_network_fully_connected();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "remove_leading_trailing_whitespaces"))
    {
        test_remove_leading_trailing_whitespaces();
    }
    else if (!strcmp(argv[1], "replace_in_array_by_value"))
    {
        error_code = test_replace_in_array_by_value();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "split_string"))
    {
        test_split_string();
    }
    else if (!strcmp(argv[1], "visualize_channel"))
    {
        error_code = test_visualize_channel();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "visualize_context_portions"))
    {
        error_code = test_visualize_context_portions();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else if (!strcmp(argv[1], "visualize_thresholded_channel"))
    {
        error_code = test_visualize_thresholded_channel();
        if (error_code < 0)
        {
            return -1;
        }
    }
    else
    {
        std::cerr << argv[1] << " is not a function to be tested." << std::endl;
        std::cerr << "The program ends." << std::endl;
        return -1;
    }
    return 0;
}


