#ifndef TESTS_H
#define TESTS_H

#include <cmath>

#include "extraction_context.h"
#include "integration_prediction_neural_network.h"
#include "interface_c_python.h"
#include "tools.h"
#include "visualization_debugging.h"

/** @brief Tests the function `append_sys_path` in the file "interface_c_python.h".
  *
  * @details The test is successful if `path_to_additional_directory` is at
  *          the beginning of the Python sys path.
  *
  * @param path_to_additional_directory Path to be appended to the Python sys path list.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_append_sys_path(const std::string& path_to_additional_directory);

/** @brief Tests the function `check_array_sorted_strictly_ascending` in the file "tools.h".
  *
  * @details The test is successful if the array is not sorted in strictly
  *          ascending order.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_check_array_sorted_strictly_ascending();

/** @brief Tests the function `check_overlapping_intra_pattern_context_portions` in the file "visualization_debugging.h".
  *
  * @details The test is successful if no error occurs during the checking
  *          of the overlapping between the intra pattern of the target patch
  *          and the two masked context portions.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_check_overlapping_intra_pattern_context_portions();

/** @brief Tests the function `create_tensors_context_portion` in the file "integration_prediction_neural_network.h".
  *
  * @details The test is successful if the six tensors have four dimensions, the
  *          1st and the 4th dimensions being equal to 1.
  *
  */
void test_create_tensors_context_portion();

/** @brief Tests the function `create_tensors_flattened_context` in the file "integration_prediction_neural_network.h".
  *
  * @details The test is successful if the two tensors have two dimensions, the
  *          1st dimension being equal to 1.
  *
  */
void test_create_tensors_flattened_context();

/** @brief Tests the function `extract_context_portions` in the file "extraction_context.h" when the width of the TB belongs to {4, 8}.
  *
  * @details The test is successful if the buffer of the flattened masked
  *          context filled by the function is always identical to the expected
  *          buffer.
  *
  * @param uiTuWidthHeight Width of the TB in pixels. The height of the TB
  *                        is equal to its width.
  * @return Error code. It is equal to -1 if an error occurs. It is equal to 0 otherwise.
  *
  */
int test_extract_context_portions_4_8(const int& uiTuWidthHeight);

/** @brief Tests the function `extract_context_portions` in the file "extraction_context.h" when the width of the TB is equal to 16 pixels.
  *
  * @details The test is successful if the buffer of the masked context
  *          portion located above the current TB filled by the function
  *          is always identical to the expected buffer. Besides, the
  *          buffer of the masked context portion located on the left side
  *          of the current TB filled by the function must always be identical
  *          to the expected buffer.
  *
  * @return Error code. It is equal to -1 if an error occurs. It is equal to 0 otherwise.
  *
  */
int test_extract_context_portions_16();

/** @brief Tests the function `fill_map_intra_prediction_modes` in the file "visualization_debugging.h".
  *
  * @details The test is successful if, in the image at "pseudo_visualization/fill_map_intra_prediction_modes.pgm",
  *          the map of intra prediction modes in white contains a 64x64 CTB in bright
  *          gray. Besides, the CTB contains a 16x16 CB storing 4 8x8 PBs, each in a
  *          different dark gray.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_fill_map_intra_prediction_modes();

/** @brief Tests the function `find_position_in_array` in the file "tools.h".
  *
  * @details The test is successful if the position of the given value
  *          in the array is 2.
  *
  */
void test_find_position_in_array();

/** @brief Tests the function `get_callable` in the file "interface_c_python.h".
  *
  * @details The test is successful if no error occurs while getting
  *          the function `load_via_pickle` in the file "loading.py".
  *
  * @param path_to_additional_directory Path to the directory containing the file "loading.py".
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_get_callable(const std::string& path_to_additional_directory);

/** @brief Tests the function `is_string_special_characters_exclusively` in the file "tools.h".
  *
  * @details The test is successful if the given string only contains special characters.
  *
  */
void test_is_string_special_characters_exclusively();

/** @brief Tests the function `load_graph` in the file "integration_prediction_neural_network.h".
  *
  * @details The test is successful if no error occurs while loading the graph.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_load_graph();

/** @brief Tests the function `load_graphs` in the file "integration_prediction_neural_network.h".
  *
  * @details The test is successful if no error occurs while loading the graphs.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_load_graphs();

/** @brief Tests the function `load_via_pickle` in the file "interface_c_python.h".
  *
  * @details The test is successful if the loaded integer is -1.
  *
  * @param path_to_additional_directory Path to the directory containing the file "loading.py".
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_load_via_pickle(const std::string& path_to_additional_directory);

/** @brief Tests the function `parse_file_strings_one_key` in the file "tools.h".
  *
  * @details The test is successful if, for each string in the file at "pseudo_data/pseudo_file_strings_one_key.txt",
  *          the string and its keys are printed together.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_parse_file_strings_one_key();

/** @brief Tests the function `parse_file_strings_three_keys` in the file "tools.h".
  *
  * @details The test is successful if, for each string in the file at "pseudo_data/pseudo_file_strings_three_keys.txt",
  *          the string and its pair of keys are inserted into the right map.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_parse_file_strings_three_keys();

/** @brief Tests a convolutional prediction neural network.
  *
  * @details The test is successful if the 2nd column of the
  *          output of the convolutional prediction neural network
  *          contains values relatively close to 20.0.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_prediction_neural_network_convolutional();

/** @brief Tests a fully-connected prediction neural network.
  *
  * @details The test is successful if the 1st, the 3rd and the 4th
  *          columns of the output of the fully-connected prediction
  *          neural network contain values close to -100.0. The 2nd
  *          column of this output contain values close to 0.0.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_prediction_neural_network_fully_connected();

/** @brief Tests the function `remove_leading_trailing_whitespaces` in the file "tools.h".
  *
  * @details The test is successful if the 1st string and the 3rd string
  *          are empty after removing their leading and trailing whitespaces.
  *
  */
void test_remove_leading_trailing_whitespaces();

/** @brief Tests the function `replace_in_array_by_value` in the file "tools.h".
  *
  * @details The test is successful if the value -11 in the first
  *          array is replaced by the value in the second array at the
  *          same position.
  *
  */
int test_replace_in_array_by_value();

/** @brief Tests the function `split_string` in the file "tools.h".
  *
  * @details The test is successful if the string is correctly split
  *          into substrings using the given set of delimiters.
  *
  */
void test_split_string();

/** @brief Tests the function `visualize_channel` in the file "visualization_debugging.h".
  *
  * @details The test is successful if the image saved at "pseudo_visualization/visualize_channel.pgm"
  *          contains a bright gray rectangle with its 2nd column in
  *          dark gray.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_visualize_channel();

/** @brief Tests the function `visualize_context_portions` in the file "visualization_debugging.h".
  *
  * @details The test is successful if, in the image saved at "pseudo_visualization/visualize_context_portions.pgm",
  *          the context portion located above the target patch is in bright
  *          gray and the context portion located on the left side of the target
  *          patch is in dark gray.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_visualize_context_portions();

/** @brief Tests the function `visualize_thresholded_channel` in the file "visualization_debugging.h".
  *
  * @details The test is successful if, in the image saved at "pseudo_visualization/visualize_thresholded_channel.ppm",
  *          the background is red, the small square is orange, the medium
  *          square is blue and the large rectangle is green.
  *
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int test_visualize_thresholded_channel();

#endif


