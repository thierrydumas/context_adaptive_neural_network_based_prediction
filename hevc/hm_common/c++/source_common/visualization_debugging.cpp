#include "visualization_debugging.h"

int check_overlapping_intra_pattern_context_portions(const float* const ptr_portion_above,
                                                     const float* const ptr_portion_left,
                                                     const int* const ptr_intra_pattern,
                                                     const unsigned int& width_target,
                                                     const float& mean_training)
{
    if (!ptr_portion_above)
    {
        fprintf(stderr, "`ptr_portion_above` is NULL.\n");
        return -1;
    }
    if (!ptr_portion_left)
    {
        fprintf(stderr, "`ptr_portion_left` is NULL.\n");
        return -1;
    }
    if (!ptr_intra_pattern)
    {
        fprintf(stderr, "`ptr_intra_pattern` is NULL.\n");
        return -1;
    }
    const float minimum(0.);
    const float maximum(255.);
    
    /*
    When `i` is equal to `width_target - 1`, the last row
    of the masked context portion located above the target
    patch is scanned.
    */
    unsigned int i(width_target - 1);
    unsigned int j(0);
    for (j = width_target - 1; j < 2*width_target; j++)
    {
        if (ptr_intra_pattern[j - width_target + 1] != static_cast<int>(std::round(std::max(minimum, std::min(maximum, ptr_portion_above[i*3*width_target + j] + mean_training)))))
        {
            fprintf(stderr, "The overlap between the masked context portion located above the target patch and the intra pattern is not correct.\n");
            return -1;
        }
    }
    
    /*
    When `j` is equal to `width_target - 1`, the last column
    of the masked context portion located on the left side of
    the target patch is scanned.
    */
    j = width_target - 1;
    for (i = 0; i < width_target; i++)
    {
        if (ptr_intra_pattern[(i + 1)*(2*width_target + 1)] != static_cast<int>(std::round(std::max(minimum, std::min(maximum, ptr_portion_left[i*width_target + j] + mean_training)))))
        {
            fprintf(stderr, "The overlap between the masked context portion located on the left side of the target patch and the intra pattern is not correct.\n");
            return -1;
        }
    }
    return 0;
}

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
                                    const unsigned int& offset_pu_in_cu_units)
{
    if (!ptr_map)
    {
        fprintf(stderr, "`ptr_map` is NULL.\n");
        return -1;
    }
    if (!ptr_intra_prediction_modes)
    {
        fprintf(stderr, "`ptr_intra_prediction_modes` is NULL.\n");
        return -1;
    }
    if (!ptr_height_cus)
    {
        fprintf(stderr, "`ptr_height_cus` is NULL.\n");
        return -1;
    }
    if (!ptr_width_cus)
    {
        fprintf(stderr, "`ptr_width_cus` is NULL.\n");
        return -1;
    }
    
    /*
    The minimum CB height is equal to 8 pixels. The minimum CB
    width is also equal to 8 pixels.
    */
    if (row_cu_in_map > height_map - 8)
    {
        fprintf(stderr, "`row_cu_in_map` is strictly larger than `height_map - 8`.\n");
        return -1;
    }
    if (column_cu_in_map > width_map - 8)
    {
        fprintf(stderr, "`column_cu_in_map` is strictly larger than `width_map - 8`.\n");
        return -1;
    }
    const unsigned char height_cu(ptr_height_cus[address_cu_in_ctu_units]);
    const unsigned char width_cu(ptr_width_cus[address_cu_in_ctu_units]);
    
    /*
    The initializations of `height_pu` and `width_pu`
    are meaningless.
    */
    unsigned char height_pu(0);
    unsigned char width_pu(0);
    if (nb_pus_in_cu == 1)
    {
        height_pu = height_cu;
        width_pu = width_cu;
    }
    else if (nb_pus_in_cu == 4)
    {
        height_pu = height_cu/2;
        width_pu = width_cu/2;
    }
    else
    {
        fprintf(stderr, "`nb_pus_in_cu` is equal to neither 1 nor 4.\n");
        return -1;
    }
    
    /*
    `ptr_cu_map` is a pointer to the pixel at the top-left
    of the current CB in the map of intra prediction modes.
    */
    unsigned char* const ptr_cu_map(ptr_map + row_cu_in_map*width_map + column_cu_in_map);
    unsigned char* ptr_pu_map(NULL);
    
    // The initialization of `intra_mode` is meaningless.
    unsigned char intra_mode(0);
    unsigned int i(0);
    unsigned char j(0);
    unsigned char k(0);
    for (i = 0; i < nb_pus_in_cu; i++)
    {
        /*
        `ptr_pu_map` is a pointer to the pixel at the top-left
        of the ith PB in the current CB.
        */
        ptr_pu_map = ptr_cu_map + (i/2)*height_pu*width_map + (i % 2)*width_pu;
        intra_mode = ptr_intra_prediction_modes[address_cu_in_ctu_units + i*offset_pu_in_cu_units];
        for (j = 0; j < height_pu; j++)
        {
            for (k = 0; k < width_pu; k++)
            {
                ptr_pu_map[j*width_map + k] = intra_mode;
            }
        }
    }
    return 0;
}

int visualize_context_portions(const float* const ptr_portion_above,
                               const float* const ptr_portion_left,
                               const unsigned int& width_target,
                               const float& mean_training,
                               const std::string& path)
{
    if (!ptr_portion_above)
    {
        fprintf(stderr, "`ptr_portion_above` is NULL.\n");
        return -1;
    }
    if (!ptr_portion_left)
    {
        fprintf(stderr, "`ptr_portion_left` is NULL.\n");
        return -1;
    }
    FILE* ptr_file(NULL);
    ptr_file = fopen(path.c_str(), "wb");
    if (!ptr_file)
    {
        fprintf(stderr, "The file at \"%s\" cannot be opened.\n", path.c_str());
        return -1;
    }
    
    /*
    `width_portion_above` is the width of the masked context
    portion located above the target patch.
    */
    const unsigned int width_portion_above(3*width_target);
    fprintf(ptr_file,
            "P5\n%u %u %u\n",
            width_portion_above,
            width_portion_above,
            255);
    const float minimum(0.);
    const float maximum(255.);
    unsigned int i(0);
    unsigned int j(0);
    for (i = 0; i < width_target; i++)
    {
        for (j = 0; j < width_portion_above; j++)
        {
            /*
            The preprocessing of the masked context, which is the
            subtraction of the mean pixels luminance from each pixel
            of the masked context, is inverted.
            */
            fputc(static_cast<int>(std::round(std::max(minimum, std::min(maximum, ptr_portion_above[i*width_portion_above + j] + mean_training)))),
                  ptr_file);
            if (ferror(ptr_file))
            {
                fclose(ptr_file);
                fprintf(stderr, "`fputc` sets the error indicator.\n");
                return -1;
            }
        }
    }
    const float* ptr_temp(ptr_portion_left);
    for (i = 0; i < 2*width_target; i++)
    {
        for (j = 0; j < width_portion_above; j++)
        {
            if (j < width_target)
            {
                fputc(static_cast<int>(std::round(std::max(minimum, std::min(maximum, ptr_temp[j] + mean_training)))),
                      ptr_file);
            }
            else
            {
                fputc(0,
                      ptr_file);
            }
            if (ferror(ptr_file))
            {
                fclose(ptr_file);
                fprintf(stderr, "`fputc` sets the error indicator.\n");
                return -1;
            }
        }
        ptr_temp += width_target;
    }
    fclose(ptr_file);
    return 0;
}


