"""A library that contains a function for creating the INRIA Holidays test set."""

import numpy
import os
import pickle
import random

import sets.untar
import tools.tools as tls

def create_holidays(source_url, path_to_directory_rgbs, path_to_holidays_set, path_to_list_indices_rotation, path_to_tar=''):
    """Creates the INRIA Holidays test set.
    
    Parameters
    ----------
    source_url : str
        URL of the original INRIA Holidays dataset.
    path_to_directory_rgbs : str
        Path to the directory to which the original INRIA Holidays
        dataset is extracted.
    path_to_holidays_set : str
        Path to the file in which the INRIA Holidays set
        is saved. The path ends with ".npy".
    path_to_list_indices_rotation : str
        Path to the file in which the list storing
        the index of each rotated YCbCr image in the
        INRIA Holidays set is saved. The path ends
        with ".pkl".
    path_to_tar : str, optional
        Path to the downloaded archive containing the original
        INRIA Holidays dataset. The default value is ''. If the
        path is not the default path, the archive is extracted
        to `path_to_directory_rgbs` before the function starts
        creating the INRIA Holidays test set.
    
    Raises
    ------
    RuntimeError
        If the INRIA Holidays set is not filled with 100
        YCbCr images.
    
    """
    if os.path.isfile(path_to_holidays_set) and os.path.isfile(path_to_list_indices_rotation):
        print('"{0}" and "{1}" already exist.'.format(path_to_holidays_set, path_to_list_indices_rotation))
        print('Delete it manually to recreate the INRIA Holidays test set.')
    else:
        if path_to_tar:
            is_downloaded = sets.untar.download_untar_archive(source_url,
                                                              path_to_directory_rgbs,
                                                              path_to_tar)
            if is_downloaded:
                print('Successfully downloaded "{}".'.format(path_to_tar))
            else:
                print('"{}" already exists.'.format(path_to_tar))
                print('Delete it manually to re-download it.')
        height_holidays = 1200
        width_holidays = 1600
        ycbcrs_uint8 = numpy.zeros((100, height_holidays, width_holidays, 3),
                                   dtype=numpy.uint8)
        
        # `list_indices_rotation` stores the index of each
        # rotated YCbCr image in the INRIA Holidays set.
        list_indices_rotation = []
        path_to_directory_jpg = os.path.join(path_to_directory_rgbs,
                                             'jpg')
        list_names = tls.clean_sort_list_strings(os.listdir(path_to_directory_jpg),
                                                 'jpg')
        
        # To make `create_holidays` deterministic, the random
        # seed should be set in the script calling `create_holidays`.
        random.shuffle(list_names)
        i = 0
        for name in list_names:
            path_to_rgb = os.path.join(path_to_directory_jpg,
                                       name)
            ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(path_to_rgb, 'RGB'))
            (height_ycbcr, width_ycbcr, _) = ycbcr_uint8.shape
            if height_ycbcr >= height_holidays and width_ycbcr >= width_holidays:
                ycbcr_potential_rotation_uint8 = ycbcr_uint8
            elif height_ycbcr >= width_holidays and width_ycbcr >= height_holidays:
                ycbcr_potential_rotation_uint8 = numpy.rot90(ycbcr_uint8)
                list_indices_rotation.append(i)
            else:
                continue
            ycbcrs_uint8[i, :, :, :] = ycbcr_potential_rotation_uint8[0:height_holidays, 0:width_holidays, :]
            i += 1
            if i == 100:
                break
        
        # If `i` is not equal to 100, the previous loop was not
        # broken. This means that at least one YCbCr image in the
        # INRIA Holidays set is filled with 0s.
        if i != 100:
            raise RuntimeError('The INRIA Holidays set is not filled with 100 YCbCr images.')
        numpy.save(path_to_holidays_set,
                   ycbcrs_uint8)
        with open(path_to_list_indices_rotation, 'wb') as file:
            pickle.dump(list_indices_rotation, file, protocol=2)


