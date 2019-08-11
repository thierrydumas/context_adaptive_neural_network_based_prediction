"""A script to create the BSDS test set."""

import argparse
import numpy
import os

import sets.setstest.bsds
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the BSDS test set.')
    parser.add_argument('path_to_directory_rgbs',
                        help='path to the directory storing the original BSDS dataset')
    parser.add_argument('--path_to_directory_tar',
                        help='path to the directory storing the downloaded archive containing the original BSDS dataset',
                        default='',
                        metavar='')
    args = parser.parse_args()
    
    path_to_bsds_set = 'sets/results/bsds/bsds.npy'
    
    # If `args.path_to_directory_tar` is equal to '', this
    # means that there is no need to download and extract
    # the archive "BSDS300-images.tgz".
    if args.path_to_directory_tar:
        path_to_tar = os.path.join(args.path_to_directory_tar,
                                   'BSDS300-images.tgz')
    else:
        path_to_tar = ''
    sets.setstest.bsds.create_bsds('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz',
                                   args.path_to_directory_rgbs,
                                   path_to_bsds_set,
                                   'sets/results/bsds/list_indices_rotation.pkl',
                                   path_to_tar=path_to_tar)
    
    # Several YCbCr images in the BSDS test set are visualized.
    bsds_uint8 = numpy.load(path_to_bsds_set)
    for i in range(4):
        tls.save_image('sets/visualization/bsds/luminance_{}.png'.format(i),
                       bsds_uint8[i, :, :, 0])
        tls.save_image('sets/visualization/bsds/chrominance_blue_{}.png'.format(i),
                       bsds_uint8[i, :, :, 1])
        tls.save_image('sets/visualization/bsds/chrominance_red_{}.png'.format(i),
                       bsds_uint8[i, :, :, 2])


