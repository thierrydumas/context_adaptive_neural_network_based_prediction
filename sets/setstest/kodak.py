"""A library that contains functions for creating the Kodak test set."""

import numpy
import os
import pickle
import six.moves.urllib.request

import tools.tools as tls

def create_kodak(url_directory, path_to_directory_rgbs, path_to_kodak_set, path_to_list_indices_rotation):
    """Creates the Kodak test set of YCbCr images.
    
    Parameters
    ----------
    url_directory : str
        URL of the directory that contains the
        24 Kodak RGB images to be downloaded.
    path_to_directory_rgbs : str
        Path to the directory in which the downloaded
        Kodak RGB images are saved.
    path_to_kodak_set : str
        Path to the file in which the Kodak test set is saved.
        The path ends with ".npy".
    path_to_list_indices_rotation : str
        Path to the file in which the list storing
        the index of each rotated YCbCr image in the
        Kodak set is saved. The path ends with ".pkl".
    
    Raises
    ------
    ValueError
        If the shape of a RGB image is neither
        (512, 768, 3) nor (768, 512, 3).
    
    """
    if os.path.isfile(path_to_kodak_set) and os.path.isfile(path_to_list_indices_rotation):
        print('"{0}" and "{1}" already exist.'.format(path_to_kodak_set, path_to_list_indices_rotation))
        print('Delete them manually to recreate the Kodak test set.')
    else:
        download_kodak(url_directory,
                       path_to_directory_rgbs)
        height_kodak = 512
        width_kodak = 768
        ycbcrs_uint8 = numpy.zeros((24, height_kodak, width_kodak, 3),
                                   dtype=numpy.uint8)
        
        # `list_indices_rotation` stores the index of
        # each rotated YCbCr image in the Kodak set.
        list_indices_rotation = []
        for i in range(24):
            path_to_rgb = os.path.join(path_to_directory_rgbs,
                                       'kodim' + str(i + 1).rjust(2, '0') + '.png')
            ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(path_to_rgb, 'RGB'))
            (height_ycbcr, width_ycbcr, _) = ycbcr_uint8.shape
            if height_ycbcr == height_kodak and width_ycbcr == width_kodak:
                ycbcrs_uint8[i, :, :, :] = ycbcr_uint8
            elif height_ycbcr == width_kodak and width_ycbcr == height_kodak:
                ycbcrs_uint8[i, :, :, :] = numpy.rot90(ycbcr_uint8)
                list_indices_rotation.append(i)
            else:
                raise ValueError('The shape of the RGB image at "{0}" is neither ({1}, {2}, 3) nor ({2}, {1}, 3).'.format(path_to_rgb, height_kodak, width_kodak))
        
        numpy.save(path_to_kodak_set,
                   ycbcrs_uint8)
        with open(path_to_list_indices_rotation, 'wb') as file:
            pickle.dump(list_indices_rotation, file, protocol=2)

def download_kodak(url_directory, path_to_directory_rgbs):
    """Downloads the 24 Kodak RGB images.
    
    Parameters
    ----------
    url_directory : str
        URL of the directory that contains the
        24 Kodak RGB images to be downloaded.
    path_to_directory_rgbs : str
        Path to the directory in which the downloaded
        Kodak RGB images are saved.
    
    """
    for i in range(24):
        filename = 'kodim' + str(i + 1).rjust(2, '0') + '.png'
        path_to_rgb = os.path.join(path_to_directory_rgbs,
                                   filename)
        if os.path.isfile(path_to_rgb):
            print('"{}" already exists.'.format(path_to_rgb))
        else:
            six.moves.urllib.request.urlretrieve(os.path.join(url_directory, filename),
                                                 path_to_rgb)
            print('Successfully downloaded "{}".'.format(filename))


