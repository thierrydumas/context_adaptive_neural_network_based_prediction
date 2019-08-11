"""A script to create the INRIA Holidays test set."""

import argparse
import numpy
import os
import random

import sets.setstest.holidays
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the INRIA Holidays test set.')
    parser.add_argument('path_to_directory_rgbs',
                        help='path to the directory storing the original INRIA Holidays dataset')
    parser.add_argument('--path_to_directory_tar',
                        help='path to the directory storing the downloaded archive containing the original INRIA Holidays dataset',
                        default='',
                        metavar='')
    args = parser.parse_args()
    
    # The random seed is set as a function called by `sets.holidays.create_holidays`
    # involves a shuffling.
    random.seed(0)
    path_to_holidays_set = 'sets/results/holidays/holidays.npy'
    
    # If `args.path_to_directory_tar` is equal to '', this
    # means that there is no need to download and extract
    # the archive "jpg1.tar.gz".
    if args.path_to_directory_tar:
        path_to_tar = os.path.join(args.path_to_directory_tar,
                                   'jpg1.tar.gz')
    else:
        path_to_tar = ''
    sets.setstest.holidays.create_holidays('ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz',
                                           args.path_to_directory_rgbs,
                                           path_to_holidays_set,
                                           'sets/results/holidays/list_indices_rotation.pkl',
                                           path_to_tar=path_to_tar)
    
    # Several YCbCr images in the INRIA Holidays set are visualized.
    holidays_uint8 = numpy.load(path_to_holidays_set)
    for i in range(4):
        tls.save_image('sets/visualization/holidays/luminance_{}.png'.format(i),
                       holidays_uint8[i, :, :, 0])
        tls.save_image('sets/visualization/holidays/chrominance_blue_{}.png'.format(i),
                       holidays_uint8[i, :, :, 1])
        tls.save_image('sets/visualization/holidays/chrominance_red_{}.png'.format(i),
                       holidays_uint8[i, :, :, 2])


