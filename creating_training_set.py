"""A script to create a training set.

The 2nd argument of this script is `path_to_directory_training_sets`,
the path to the directory storing the different training sets. To order
the different training sets, the following directory tree is created:

path_to_directory_training_sets/*/**/***/****/

*: either "width_target_4" or "width_target_8" or "width_target_None".
**: either "single" or "pair".
***: either "luminance" or "chrominance_blue" or "chrominance_red".
****: "thread_i" with i depending on the number of threads used.

"""

import argparse
import numpy
import os
import pickle
import random
import tensorflow as tf
import time

import hevc.constants
import parsing.parsing
import sets.reading
import sets.writing
import tools.tools as tls

def check_tfrecord_without_extraction(path_to_tfrecord, is_pair, path_to_directory_vis):
    """Checks the content of a file ".tfrecord" in the training set.
    
    `check_tfrecord_without_extraction` is valid only when
    the training set is built without extraction of target
    patches, each paired with its two context portions.
    
    Parameters
    ----------
    path_to_tfrecord : str
        Path to the file ".tfrecord".
    is_pair : bool
        Is each image channel in the training set paired
        with its version with HEVC compression artifacts?
    path_to_directory_vis : str
        Path to the directory storing the visualizations
        of the content of the file ".tfrecord".
    
    """
    queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                    shuffle=False)
    reader = tf.TFRecordReader()
    serialized_example = reader.read(queue_tfrecord)[1]
    node_channel_single_or_pair_uint8 = sets.reading.parse_example_channel_single_or_pair(serialized_example,
                                                                                          is_pair)
    coordinator = tf.train.Coordinator()
    with tf.Session() as sess:
        list_threads = tf.train.start_queue_runners(coord=coordinator)
        
        # The file at `path_to_tfrecord` must contain at least
        # four examples.
        for i in range(4):
            channel_single_or_pair_uint8 = sess.run(node_channel_single_or_pair_uint8)
            if is_pair:
                tls.save_image(os.path.join(path_to_directory_vis, 'example_original_{}.png'.format(i)),
                               channel_single_or_pair_uint8[:, :, 0])
                tls.save_image(os.path.join(path_to_directory_vis, 'example_artifacts_{}.png'.format(i)),
                               channel_single_or_pair_uint8[:, :, 1])
            else:
                tls.save_image(os.path.join(path_to_directory_vis, 'example_{}.png'.format(i)),
                               numpy.squeeze(channel_single_or_pair_uint8, axis=2))
        coordinator.request_stop()
        coordinator.join(list_threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a training set.')
    parser.add_argument('path_to_directory_extraction',
                        help='path to the directory to which "ILSVRC2012_img_train.tar" was extracted')
    parser.add_argument('path_to_directory_training_sets',
                        help='path to the directory storing the different training sets')
    parser.add_argument('width_target',
                        help='width of the target patch, None indicating that there is no extraction of target patches, each paired with its two context portions')
    parser.add_argument('index_channel',
                        help='channel index',
                        type=parsing.parsing.int_positive)
    parser.add_argument('--nb_threads',
                        help='number of threads for creating the files ".tfrecord"',
                        type=parsing.parsing.int_strictly_positive,
                        default=4,
                        metavar='')
    parser.add_argument('--nb_rgbs_per_tfrecord',
                        help='maximum number of RGB images that are preprocessed to fill a file ".tfrecord"',
                        type=parsing.parsing.int_strictly_positive,
                        default=1000,
                        metavar='')
    parser.add_argument('--is_pair',
                        help='if given, the contexts in the training set contain HEVC compression artifacts',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    
    tag_width_target = 'width_target_{}'.format(args.width_target)
    if args.is_pair:
        tag_pair = 'pair'
    else:
        tag_pair = 'single'
    
    # `sets.writing.create_tfrecord_threading` checks that
    # `args.index_channel` belongs to {0, 1, 2}.
    if args.index_channel == 0:
        tag_channel = 'luminance'
    elif args.index_channel == 1:
        tag_channel = 'chrominance_blue'
    else:
        tag_channel = 'chrominance_red'
    if args.width_target == 'None':
        width_target = None
    else:
        width_target = int(args.width_target)
    if args.is_pair:
        path_to_directory_temp = os.path.join('hevc/temp/creating_training_set/',
                                              tag_width_target,
                                              tag_channel)
        if not os.path.isdir(path_to_directory_temp):
            os.makedirs(path_to_directory_temp)
        
        # Many different training sets can be created in parallel
        # without mixing their associated input file to HEVC and
        # output file.
        dict_pair_threads = {
            'paths_to_before_encoding_hevc': [
                os.path.join(path_to_directory_temp, 'image_before_encoding_hevc_{}.yuv'.format(i)) for i in range(args.nb_threads)
            ],
            'paths_to_after_encoding_hevc': [
                os.path.join(path_to_directory_temp, 'image_after_encoding_hevc_{}.yuv'.format(i)) for i in range(args.nb_threads)
            ],
            'path_to_cfg': 'hevc/configuration/intra_main.cfg',
            'paths_to_bitstream': [
                os.path.join(path_to_directory_temp, 'bitstream_{}.bin'.format(i)) for i in range(args.nb_threads)
            ],
            'path_to_exe_encoder': hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
            'qps_int': numpy.array([32, 37, 42],
                                   dtype=numpy.int32)
        }
    else:
        dict_pair_threads = None
    
    # The directory tree in the directory at `args.path_to_directory_training_sets`
    # has a specific structure (see the documentation of this script).
    path_to_directory_threads = os.path.join(args.path_to_directory_training_sets,
                                             tag_width_target,
                                             tag_pair,
                                             tag_channel)
    paths_to_directories_tfrecords = []
    for i in range(args.nb_threads):
        path_to_directory_tfrecord = os.path.join(path_to_directory_threads,
                                                  'thread_{}'.format(i))
        paths_to_directories_tfrecords.append(path_to_directory_tfrecord)
        if not os.path.isdir(path_to_directory_tfrecord):
            os.makedirs(path_to_directory_tfrecord)
    
    t_start = time.time()
    paths_to_rgbs = tls.collect_paths_to_files_in_subdirectories(args.path_to_directory_extraction,
                                                                 ('.JPEG', '.jpg'))
    
    # The images are shuffled to break the arrangement
    # per class.
    random.shuffle(paths_to_rgbs)
    sets.writing.create_tfrecord_threading(paths_to_directories_tfrecords,
                                           paths_to_rgbs,
                                           width_target,
                                           args.index_channel,
                                           args.nb_rgbs_per_tfrecord,
                                           dict_pair_threads)
    t_stop = time.time()
    nb_hours = int((t_stop - t_start)/3600)
    nb_minutes = int((t_stop - t_start)/60)
    print('\nThe creation of the training set toke {0} hours and {1} minutes.'.format(nb_hours, nb_minutes - 60*nb_hours))
    
    # The training set is checked.
    if args.width_target == 'None':
        path_to_directory_vis = os.path.join('sets/visualization/training_set/',
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        check_tfrecord_without_extraction(os.path.join(path_to_directory_threads, 'thread_0', 'data_0.tfrecord'),
                                          args.is_pair,
                                          path_to_directory_vis)
    
    # If the directory at `path_to_directory_nb_examples`
    # does not exist, it is created.
    path_to_directory_nb_examples = os.path.join('sets/results/training_set/count/',
                                                 tag_pair,
                                                 tag_channel)
    if not os.path.isdir(path_to_directory_nb_examples):
        os.makedirs(path_to_directory_nb_examples)
    path_to_nb_examples = os.path.join(path_to_directory_nb_examples,
                                       'nb_examples_{}.pkl'.format(tag_width_target))
    if os.path.isfile(path_to_nb_examples):
        print('The file at "{}" already exists.'.format(path_to_nb_examples))
    else:
        
        # The number of examples in the newly created training
        # set is counted.
        nb_examples = sets.writing.count_examples_in_files_tfrecord(path_to_directory_threads)
        with open(path_to_nb_examples, 'wb') as file:
            pickle.dump(nb_examples, file, protocol=2)


