"""A library that contains a function for creating the BSDS test set."""

import numpy
import os
import pickle

import sets.untar
import tools.tools as tls

def create_bsds(source_url, path_to_directory_rgbs, path_to_bsds_set, path_to_list_indices_rotation, path_to_tar=''):
    """Creates the BSDS test set of YCbCr images.
    
    Parameters
    ----------
    source_url : str
        URL of the original BSDS dataset.
    path_to_directory_rgbs : str
        Path to the directory to which the original BSDS dataset
        (training RGB images and test RGB images) is extracted.
    path_to_bsds_set : str
        Path to the file in which the BSDS test set is saved.
        The path ends with ".npy".
    path_to_list_indices_rotation : str
        Path to the file in which the list storing
        the index of each rotated YCbCr image in the
        BSDS test set is saved. The path ends with ".pkl".
    path_to_tar : str, optional
        Path to the downloaded archive containing the original
        BSDS dataset. The default value is ''. If the path
        is not the default path, the archive is extracted
        to `path_to_directory_rgbs` before the function starts
        creating the BSDS test set.
    
    Raises
    ------
    RuntimeError
        The number of BSDS RGB images to be read is not 100.
    ValueError
        If the shape of a RGB image in the directory
        at `path_to_directory_rgbs` is neither (321, 481, 3)
        nor (481, 321, 3).
    
    """
    if os.path.isfile(path_to_bsds_set) and os.path.isfile(path_to_list_indices_rotation):
        print('"{0}" and "{1}" already exist.'.format(path_to_bsds_set, path_to_list_indices_rotation))
        print('Delete them manually to recreate the BSDS test set.')
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
        height_bsds = 321
        width_bsds = 481
        ycbcrs_uint8 = numpy.zeros((100, height_bsds - 1, width_bsds - 1, 3),
                                   dtype=numpy.uint8)
        
        # `list_indices_rotation` stores the index of each
        # rotated YCbCr image in the BSDS test set.
        list_indices_rotation = []
        path_to_directory_test = os.path.join(path_to_directory_rgbs,
                                              'BSDS300/images/test/')
        list_names = tls.clean_sort_list_strings(os.listdir(path_to_directory_test),
                                                 'jpg')
        if len(list_names) != 100:
            raise RuntimeError('The number of BSDS RGB images to be read is not 100.')
        for i in range(100):
            path_to_rgb = os.path.join(path_to_directory_test,
                                       list_names[i])
            ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(path_to_rgb, 'RGB'))
            (height_ycbcr, width_ycbcr, _) = ycbcr_uint8.shape
            if height_ycbcr == height_bsds and width_ycbcr == width_bsds:
                ycbcr_potential_rotation_uint8 = ycbcr_uint8
            elif height_ycbcr == width_bsds and width_ycbcr == height_bsds:
                ycbcr_potential_rotation_uint8 = numpy.rot90(ycbcr_uint8)
                list_indices_rotation.append(i)
            else:
                raise ValueError('The shape of the RGB image at "{0}" is neither ({1}, {2}, 3) nor ({2}, {1}, 3).'.format(path_to_rgb, height_bsds, width_bsds))
            ycbcrs_uint8[i, :, :, :] = ycbcr_potential_rotation_uint8[0:height_bsds - 1, 0:width_bsds - 1, :]
        
        numpy.save(path_to_bsds_set,
                   ycbcrs_uint8)
        with open(path_to_list_indices_rotation, 'wb') as file:
            pickle.dump(list_indices_rotation, file, protocol=2)


