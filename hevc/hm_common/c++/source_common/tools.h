#ifndef TOOLS_H
#define TOOLS_H

#include <fstream>
#include <iostream>
#include <map>
#include <regex>

/** @brief Checks whether the array is sorted in strictly ascending order.
  *
  * @param ptr_array Pointer to the array to be checked.
  * @param size_array Size of the array to be checked.
  * @param is_sorted_strictly_ascending True if the array is sorted in strictly
  *                                     ascending order.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
template <typename T>
int check_array_sorted_strictly_ascending(const T* const ptr_array,
                                          const unsigned int& size_array,
                                          bool& is_sorted_strictly_ascending)
{
    if (!ptr_array)
    {
        fprintf(stderr, "`ptr_array` is NULL.");
        return -1;
    }
    if (size_array == 1 || size_array == 0)
    {
        is_sorted_strictly_ascending = true;
        return 0;
    }
    if (ptr_array[size_array - 1] <= ptr_array[size_array - 2])
    {
        is_sorted_strictly_ascending = false;
        return 0;
    }
    return check_array_sorted_strictly_ascending<T>(ptr_array,
                                                    size_array - 1,
                                                    is_sorted_strictly_ascending);
}

/** @brief Finds the position of a given value in the array.
  *
  *  @param ptr_array Pointer to the array buffer.
  *  @param size_array Number of elements in the array.
  *  @param value Given value.
  *  @return Position of the given value in the array. -1 is
  *          returned if the given value is not in the array.
  */
template <typename T>
int find_position_in_array(const T* const ptr_array,
                           const unsigned int& size_array,
                           const T& value)
{
   if (!ptr_array)
    {
        std::cerr << "`ptr_array` is NULL." << std::endl;
        return -1;
    }
    for (unsigned int i(0); i < size_array; i++)
    {
        if (ptr_array[i] == value)
        {
            return i;
        }
    }
    return -1;
}

/** @brief Checks whether the given string only contains special characters.
  *
  * @param string_in Given string to be checked.
  * @return Does the given string contain special characters exclusively?
  *
  */
bool is_string_special_characters_exclusively(const std::string& string_in);

/** @brief Parses the file in which each line contains one unsigned integer key and a string.
  *
  * @param map_storage Map associating each unsigned integer key to the string.
  * @param path_to_file Path to the file to be parsed.
  * @param delimiters Set of delimiters for delimiting a key and a string.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int parse_file_strings_one_key(std::map<unsigned int, std::string>& map_storage,
                               const std::string& path_to_file,
                               const std::string& delimiters);

/** @brief Parses a file in which each line contains three keys and a string.
  *
  * @details The three keys are: an unsigned integer, a boolean and an unsigned integer.
  *          They identify the string.
  *
  * @param map_false Map associating each pair of unsigned integer keys to the
  *                  string. The boolean key is false.
  * @param map_true Map associating each pair of unsigned integer keys to the
  *                 string. The boolean key is true.
  * @param path_to_file Path to the file to be parsed.
  * @param delimiters Set of delimiters for delimiting two keys or a key and a string.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int parse_file_strings_three_keys(std::map<std::pair<unsigned int, unsigned int>, std::string>& map_false,
                                  std::map<std::pair<unsigned int, unsigned int>, std::string>& map_true,
                                  const std::string& path_to_file,
                                  const std::string& delimiters);

/** @brief Removes all the leading and trailing whitespaces from a string.
  *
  * @param string_to_be_modified String whose leading and trailing whitespaces
  *                              are to be removed.
  *
  */                      
void remove_leading_trailing_whitespaces(std::string& string_to_be_modified);


/** @brief Replaces a value in an array by the value in another array at the same position.
  *
  * @param ptr_array_replacement Pointer to the array in which the value
  *                              is to be replaced.
  * @param ptr_array_reference Pointer to the reference array containing the
  *                            values for the replacement.
  * @param size_arrays Size of the two arrays.
  * @param value_replacement Value to be replaced.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
template <typename T>
int replace_in_array_by_value(T* const ptr_array_replacement,
                              const T* const ptr_array_reference,
                              const unsigned int& size_arrays,
                              const T& value_replacement)
{
    if (!ptr_array_replacement)
    {
        fprintf(stderr, "`ptr_array_replacement` is NULL.\n");
        return -1;
    }
    if (!ptr_array_reference)
    {
        fprintf(stderr, "`ptr_array_reference` is NULL.\n");
        return -1;
    }
    for (unsigned int i(0); i < size_arrays; i++)
    {
        if (ptr_array_replacement[i] == value_replacement)
        {
            ptr_array_replacement[i] = ptr_array_reference[i];
        }
    }
    return 0;
}

/** @brief Splits a string into substrings using a given set of delimiters.
  *
  * @param vector_substrings Vector storing the substrings resulting from
  *                          the split.
  * @param input_string Input string to be split into substrings.
  * @param delimiters Set of delimiters.
  *
  */
void split_string(std::vector<std::string>& vector_substrings,
                  const std::string& input_string,
                  const std::string& delimiters);

#endif


