"""A library containing functions for loading either luminance images or grayscale images from files."""

import glob
import numpy
import os

import hevc.running
import tools.tools as tls

DICTIONARY_DESCRIPTION = {
    'A_PeopleOnStreet': (1600, 2560),
    'A_Traffic': (1600, 2560),
    'B_BasketballDrive': (1080, 1920),
    'B_BQTerrace': (1080, 1920),
    'B_Cactus': (1080, 1920),
    'B_Kimono': (1080, 1920),
    'B_ParkScene': (1080, 1920),
    'C_BasketballDrill': (480, 832),
    'C_BQMall': (480, 832),
    'C_PartyScene': (480, 832),
    'C_RaceHorses': (480, 832),
    'D_BasketballPass': (240, 416),
    'D_BQSquare': (240, 416),
    'D_BlowingBubbles': (240, 416),
    'D_RaceHorses': (240, 416),
    'Bus': (288, 352),
    'City': (576, 704),
    'Crew': (576, 704),
    'Football': (288, 352),
    'Foreman': (288, 352),
    'Harbour': (576, 704),
    'Mobile': (288, 352),
    'Soccer': (576, 704),
    
    # The video named 'video_ice' is used for testing exclusively.
    'video_ice': (288, 352)
}

# The functions are sorted in alphabetic order.

def find_load_grayscale(path_to_directory_file, prefix_filename):
    """Finds and loads a grayscale image.
    
    Parameters
    ----------
    path_to_directory_file : str
        Path to the directory storing the grayscale image.
    prefix_filename : str
        Prefix of the name of the grayscale image file.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Loaded grayscale image. The 3rd array dimension is equal to 1.
    
    Raises
    ------
    IOError
        If multiple files ".jpg" or ".png" whose names start with `prefix_filename`
        exist in the directory at `path_to_directory_file`.
    
    """
    # `glob.glob` returns a list of strings.
    paths_found = glob.glob(os.path.join(path_to_directory_file, '{}*.png'.format(prefix_filename)))
    paths_found += glob.glob(os.path.join(path_to_directory_file, '{}*.jpg'.format(prefix_filename)))
    if len(paths_found) != 1:
        raise IOError('The number of files ".jpg" or ".png" whose names start with "{0}" in the directory at "{1}" is not equal to 1.'.format(prefix_filename, path_to_directory_file))
    grayscale_uint8 = tls.read_image_mode(paths_found[0],
                                          'L')
    height_divisible_8 = 8*(grayscale_uint8.shape[0]//8)
    width_divisible_8 = 8*(grayscale_uint8.shape[1]//8)
    
    # For any image with one channel, the third dimension
    # of the array storing the image is equal to 1.
    return numpy.expand_dims(grayscale_uint8[0:height_divisible_8, 0:width_divisible_8], 2)

def find_rgb_load_luminance(path_to_directory_file, prefix_filename):
    """Finds a RGB image and loads a luminance image from its content.
    
    Parameters
    ----------
    path_to_directory_file : str
        Path to the directory storing the RGB image.
    prefix_filename : str
        Prefix of the name of the RGB image file.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Loaded luminance image. The 3rd array dimension is equal to 1.
    
    Raises
    ------
    IOError
        If multiple files ".jpg" or ".png" whose names start with `prefix_filename`
        exist in the directory at `path_to_directory_file`.
    
    """
    paths_found = glob.glob(os.path.join(path_to_directory_file, '{}*.png'.format(prefix_filename)))
    paths_found += glob.glob(os.path.join(path_to_directory_file, '{}*.jpg'.format(prefix_filename)))
    if len(paths_found) != 1:
        raise IOError('The number of files ".jpg" or ".png" whose names start with "{0}" in the directory at "{1}" is not equal to 1.'.format(prefix_filename, path_to_directory_file))
    ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(paths_found[0], 'RGB'))
    height_divisible_8 = 8*(ycbcr_uint8.shape[0]//8)
    width_divisible_8 = 8*(ycbcr_uint8.shape[1]//8)
    return ycbcr_uint8[0:height_divisible_8, 0:width_divisible_8, 0:1]

def find_video_load_luminance(path_to_directory_file, prefix_filename, idx_frame=0):
    """Finds a YCbCr video and loads a luminance image from its content.
    
    Parameters
    ----------
    path_to_directory_file : str
        Path to the directory storing the YCbCr video.
    prefix_filename : str
        Prefix of the name of the YCbCr video file.
    idx_frame : int, optional
        Index of the frame to be extracted from the
        YCbCr video. The default value is 0.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Loaded luminance image. The 3rd array dimension is equal to 1.
    
    Raises
    ------
    IOError
        If multiple files ".yuv" whose names start with `prefix_filename`
        exist in the directory at `path_to_directory_file`.
    
    """
    paths_found = glob.glob(os.path.join(path_to_directory_file, '{}*.yuv'.format(prefix_filename)))
    if len(paths_found) != 1:
        raise IOError('The number of files ".yuv" whose names start with "{0}" in the directory at "{1}" is not equal to 1.'.format(prefix_filename, path_to_directory_file))
    (height_video, width_video) = DICTIONARY_DESCRIPTION[prefix_filename]
    ycbcrs_uint8 = hevc.running.read_400_or_420(height_video,
                                                width_video,
                                                idx_frame + 1,
                                                numpy.uint8,
                                                False,
                                                paths_found[0])
    
    # Only the luminance channel of the last loaded frame is kept.
    return ycbcrs_uint8[:, :, 0:1, idx_frame]


