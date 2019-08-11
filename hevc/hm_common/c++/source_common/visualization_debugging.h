#ifndef VISUALIZATION_DEBUGGING_H
#define VISUALIZATION_DEBUGGING_H

#include <stdio.h>
#include <string>

#include "tools.h"

/** @brief Checks that the intra pattern of the target patch and
  *        the two masked context portions overlap correctly.
  *
  * @param ptr_portion_above Pointer to the buffer of the masked context
  *                          portion located above the target patch. The
  *                          shape of the masked context portion located
  *                          above the target patch is (`width_target`, `3*width_target`).
  * @param ptr_portion_left Pointer to the buffer of the masked context
  *                         portion located on the left side of the target
  *                         patch. The shape of the masked context portion
  *                         located on the left side of the target patch is
  *                         (`2*width_target`, `width_target`).
  * @param ptr_intra_pattern Pointer to the buffer of the intra pattern
  *                          of the target patch. The shape of the intra
  *                          pattern is (`2*width_target + 1`, `2*width_target + 1`).
  * @param width_target Width of the target patch.
  * @param mean_training Mean pixels luminance computed over different
  *                      luminance images.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int check_overlapping_intra_pattern_context_portions(const float* const ptr_portion_above,
                                                     const float* const ptr_portion_left,
                                                     const int* const ptr_intra_pattern,
                                                     const unsigned int& width_target,
                                                     const float& mean_training);

/** @brief Fills a CB in the current CTB in the map of intra prediction
  *        modes with the intra prediction mode of each PB.
  *
  * @param ptr_map Pointer to the pixel at the top-left of the
  *                the map of intra prediction modes.
  * @param height_map Height of the map of intra prediction modes.
  * @param width_map Width of the map of intra prediction modes.
  * @param ptr_intra_prediction_modes Pointer to the 1st element in the array of
  *                                   intra prediction modes for the current CTB.
  * @param ptr_height_cus Pointer to the 1st element in the array of CBs height
  *                       for the current CTB.
  * @param ptr_width_cus Pointer to the 1st element in the array of CBs width
  *                      for the current CTB.
  * @param row_cu_in_map Row of the current CB in the map of intra prediction
  *                      modes.
  * @param column_cu_in_map Column of the current CB in the map of intra
  *                         prediction modes.
  * @param address_cu_in_ctu_units Address of the CB in the current CTB, expressed
  *                                in units. It is a Z-scan address.
  * @param nb_pus_in_cu Number of PBs in the current CB.
  * @param offset_pu_in_cu_units Offset between two PBs in the current CB, expressed
  *                              in units.
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int fill_map_intra_prediction_modes(unsigned char* const ptr_map,
                                    const unsigned int& height_map,
                                    const unsigned int& width_map,
                                    const unsigned char* const ptr_intra_prediction_modes,
                                    const unsigned char* const ptr_height_cus,
                                    const unsigned char* const ptr_width_cus,
                                    const unsigned int& row_cu_in_map,
                                    const unsigned int& column_cu_in_map,
                                    const unsigned int& address_cu_in_ctu_units,
                                    const unsigned int& nb_pus_in_cu,
                                    const unsigned int& offset_pu_in_cu_units);

/** @brief Saves an image of the channel.
  *
  * @param ptr_channel Pointer to the buffer of the channel. The elements
  *                    of the buffer have integer type.
  * @param height_channel Height of the channel.
  * @param width_channel Width of the channel.
  * @param maximum_gray_level Maximum gray level for the PGM format.
  * @param path Path to the saved image. The path ends with ".pgm".
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
template <typename T>
int visualize_channel(const T* const ptr_channel,
                      const unsigned int& height_channel,
                      const unsigned int& width_channel,
                      const unsigned int& maximum_gray_level,
                      const std::string& path)
{
    if (!ptr_channel)
    {
        fprintf(stderr, "`ptr_channel` is NULL.\n");
        return -1;
    }
    FILE* ptr_file(NULL);
    ptr_file = fopen(path.c_str(), "wb");
    if (!ptr_file)
    {
        fprintf(stderr, "The file at \"%s\" cannot be opened.\n", path.c_str());
        return -1;
    }
    
    // "P5" is the identifier for the PGM binary format.
    fprintf(ptr_file,
            "P5\n%u %u %u\n",
            width_channel,
            height_channel,
            maximum_gray_level);
    unsigned int i(0);
    unsigned int j(0);
    for (i = 0; i < height_channel; i++)
    {
        for (j = 0; j < width_channel; j++)
        {
            fputc(ptr_channel[i*width_channel + j],
                  ptr_file);
            if (ferror(ptr_file))
            {
                fclose(ptr_file);
                fprintf(stderr, "`fputc` sets the error indicator.\n");
                return -1;
            }
        }
    }
    fclose(ptr_file);
    return 0;
}

/** @brief Arranges two masked context portions in a single image and saves the image.
  *
  * @param ptr_portion_above Pointer to the buffer of the masked context
  *                          portion located above the target patch.
  * @param ptr_portion_left Pointer to the buffer of the masked context
  *                         portion located on the left side of the target
  *                         patch.
  * @param width_target Width of the target patch.
  * @param mean_training Mean pixels luminance computed over different
  *                      luminance images.
  * @param path Path to the saved image. The path ends with ".pgm".
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
int visualize_context_portions(const float* const ptr_portion_above,
                               const float* const ptr_portion_left,
                               const unsigned int& width_target,
                               const float& mean_training,
                               const std::string& path);

/** @brief Saves a RGB image of the thresholded channel.
  *
  * @param ptr_channel Pointer to the buffer of the channel. The elements
  *                     of the buffer have integer type.
  * @param height_channel Height of the channel.
  * @param width_channel Width of the channel.
  * @param ptr_array_thresholds Pointer to the array of thresholds.
  * @param size_arrays_thresholds Size of the array of thresholds.
  * @param ptr_array_colors 2D array whose ith row contains the RGB components
  *                         for the ith threshold. Its number of rows is equal
  *                         to `size_arrays_thresholds + 1`.
  * @param maximum_color_level Maximum color level for the PPM format.
  * @param path Path to the saved RGB image. The path ends with ".ppm".
  * @return Error code. It is equal to -1 if an error occurs. Otherwise, it is equal to 0.
  *
  */
template <typename T>
int visualize_thresholded_channel(const T* const ptr_channel,
                                  const unsigned int& height_channel,
                                  const unsigned int& width_channel,
                                  const T* const ptr_array_thresholds,
                                  const unsigned int& size_arrays_thresholds,
                                  const unsigned int ptr_array_colors[][3],
                                  const unsigned int& maximum_color_level,
                                  const std::string& path)
{
    if (!ptr_channel)
    {
        fprintf(stderr, "`ptr_channel` is NULL.\n");
        return -1;
    }
    if (!ptr_array_thresholds)
    {
        fprintf(stderr, "`ptr_array_thresholds` is NULL.\n");
        return -1;
    }
    
    /*
    If the array of thresholds is not sorted in strictly
    ascending order, a grayscale level in the channel can
    be associated to two colors.
    */
    bool is_sorted_strictly_ascending(false);
    const int error_code(check_array_sorted_strictly_ascending<T>(ptr_array_thresholds,
                                                                  size_arrays_thresholds,
                                                                  is_sorted_strictly_ascending));
    if (error_code < 0)
    {
        return -1;
    }
    if (!is_sorted_strictly_ascending)
    {
        fprintf(stderr, "The array of thresholds is not sorted in strictly ascending order.\n");
        return -1;
    }
    if (!ptr_array_colors)
    {
        fprintf(stderr, "`ptr_array_colors` is NULL.\n");
        return -1;
    }
    FILE* ptr_file(NULL);
    ptr_file = fopen(path.c_str(), "wb");
    if (!ptr_file)
    {
        fprintf(stderr, "The file at \"%s\" cannot be opened.\n", path.c_str());
        return -1;
    }
    
    // "P6" is the identifier for the PPM format.
    fprintf(ptr_file,
            "P6\n%u %u %u\n",
            width_channel,
            height_channel,
            maximum_color_level);
    unsigned int i(0);
    unsigned int j(0);
    unsigned int k(0);
    unsigned int index_color(0);
    bool is_color_assigned(false);
    for (i = 0; i < height_channel; i++)
    {
        for (j = 0; j < width_channel; j++)
        {
            is_color_assigned = false;
            for (k = 0; k < size_arrays_thresholds; k++)
            {
                if (ptr_channel[i*width_channel + j] <= ptr_array_thresholds[k])
                {
                    index_color = k;
                    is_color_assigned = true;
                    break;
                }
            }
            
            /*
            At this point, if `is_color_assigned` is false, `ptr_channel[i*width_channel + j]`
            is strictly larger than `ptr_array_thresholds[size_arrays_thresholds - 1]`, which
            is the largest threshold.
            */
            if (!is_color_assigned)
            {
                index_color = size_arrays_thresholds;
            }
            for (k = 0; k < 3; k++)
            {
                fputc(ptr_array_colors[index_color][k],
                      ptr_file);
                
                // If `fputc` fails, the error indicator is set.
                if (ferror(ptr_file))
                {
                    fclose(ptr_file);
                    fprintf(stderr, "`fputc` sets the error indicator.\n");
                    return -1;
                }
            }
        }
    }
    fclose(ptr_file);
    return 0;
}

#endif


