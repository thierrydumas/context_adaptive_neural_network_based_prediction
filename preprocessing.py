"""A script to create the training subset of YCbCr images and compute the mean pixel intensity for each channel Y, Cb, and Cr."""

import argparse
import numpy
import os

import parsing.parsing
import sets.writing
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the training subset and computes the mean pixel intensity for each channel Y, Cb, and Cr.')
    parser.add_argument('path_to_directory_extraction',
                        help='path to the directory to which "ILSVRC2012_img_train.tar" is extracted')
    parser.add_argument('--nb_threads',
                        help='number of threads for creating the files ".tfrecord"',
                        type=parsing.parsing.int_strictly_positive,
                        default=4,
                        metavar='')
    args = parser.parse_args()
    
    paths_to_means_training = (
        'sets/results/training_set/means/luminance/mean_training.pkl',
        'sets/results/training_set/means/chrominance_blue/mean_training.pkl',
        'sets/results/training_set/means/chrominance_red/mean_training.pkl'
    )
    path_to_training_subset = 'sets/results/training_subset/training_subset.npy'
    isfile_means = all([os.path.isfile(path_to_mean_training) for path_to_mean_training in paths_to_means_training])
    isfile_training_subset = os.path.isfile(path_to_training_subset)
    
    # `tls.collect_paths_to_files_in_subdirectories` is not called
    # if it is not necessary.
    if isfile_means and isfile_training_subset:
        print('The files at the following paths already exist.')
        print(paths_to_means_training)
        print(path_to_training_subset)
    else:
        
        # The paths in `paths_to_rgbs` are not shuffled
        # because the current script has to be deterministic.
        paths_to_rgbs = tls.collect_paths_to_files_in_subdirectories(args.path_to_directory_extraction,
                                                                     ('.JPEG', '.jpg'))
        if isfile_means:
            print('The files at the following paths already exist.')
            print(paths_to_means_training)
        else:
            means_intensities_float64 = sets.writing.compute_mean_intensities_threading(paths_to_rgbs,
                                                                                        args.nb_threads)
            
            for i in range(means_intensities_float64.size):
                with open(paths_to_means_training[i], 'wb') as file:
                    pickle.dump(means_intensities_float64[i].item(), file, protocol=2)
        
        # The number of YCbCr images in the training subset is
        # equal to the number of YCbCr images in the BSDS set.
        if isfile_training_subset:
            print('The file at "{}" already exists.'.format(path_to_training_subset))
        else:
            sets.writing.create_training_subset(paths_to_rgbs,
                                                path_to_training_subset,
                                                100)
            training_subset_uint8 = numpy.load(path_to_training_subset)
            for i in range(4):
                tls.save_image('sets/visualization/training_subset/luminance_{}.png'.format(i),
                               training_subset_uint8[i, :, :, 0])
                tls.save_image('sets/visualization/training_subset/chrominance_blue_{}.png'.format(i),
                               training_subset_uint8[i, :, :, 1])
                tls.save_image('sets/visualization/training_subset/chrominance_red_{}.png'.format(i),
                               training_subset_uint8[i, :, :, 2])


