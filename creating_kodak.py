"""A script to create the Kodak test set."""

import argparse
import numpy

import sets.setstest.kodak
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the Kodak test set.')
    parser.parse_args()
    
    path_to_kodak_set = 'sets/results/kodak/kodak.npy'
    
    sets.setstest.kodak.create_kodak('http://r0k.us/graphics/kodak/kodak/',
                                     'sets/results/kodak/rgbs/',
                                     path_to_kodak_set,
                                     'sets/results/kodak/list_indices_rotation.pkl')
    
    # Several YCbCr images in the Kodak test set are visualized.
    kodak_uint8 = numpy.load(path_to_kodak_set)
    for i in range(4):
        tls.save_image('sets/visualization/kodak/luminance_{}.png'.format(i),
                       kodak_uint8[i, :, :, 0])
        tls.save_image('sets/visualization/kodak/chrominance_blue_{}.png'.format(i),
                       kodak_uint8[i, :, :, 1])
        tls.save_image('sets/visualization/kodak/chrominance_red_{}.png'.format(i),
                       kodak_uint8[i, :, :, 2])


