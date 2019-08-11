"""A script to test all the libraries in the directory "sets"."""

import argparse
import numpy
import os

import hevc.constants

# The name of the module `Queue` has been changed
# in Python 3.x <https://www.python.org/dev/peps/pep-3108/#pep-8-violations-done>.
try:
    import queue
except ImportError:
    import Queue as queue
import tensorflow as tf

import sets.arranging
import sets.common
import sets.reading
import sets.untar
import sets.writing
import tools.tools as tls


class TesterSets(object):
    """Class for testing all the libraries in the directory "sets"."""
    
    def test_arrange_context_portions(self):
        """Tests the function `arrange_context_portions` in the file "sets/arranging.py".
        
        An image is saved at
        "sets/pseudo_visualization/arrange_context_portions.png".
        The test is successful if, in this image, the
        context portion located above the target patch
        (in black) and the context portion located on the
        left side of the target patch (in gray) form a "L"
        flipped top to bottom.
        
        """
        width_target = 32
        
        portion_above_uint8 = numpy.zeros((width_target, 3*width_target, 1), dtype=numpy.uint8)
        portion_left_uint8 = 120*numpy.ones((2*width_target, width_target, 1), dtype=numpy.uint8)
        image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                              portion_left_uint8)
        tls.save_image('sets/pseudo_visualization/arrange_context_portions.png',
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_arrange_flattened_context(self):
        """Tests the function `arrange_flattened_context` in the file "sets/arranging.py".
        
        An image is saved at
        "sets/pseudo_visualization/arrange_flattened_context.png".
        The test is successful if, in this image, the
        2nd row of the context portion located above
        the target patch is in black and the 2nd last
        row of the context portion located on the left
        side of the target patch is in dark gray.
        
        """
        width_target = 16
        
        flattened_context_uint8 = 120*numpy.ones(5*width_target**2, dtype=numpy.uint8)
        flattened_context_uint8[3*width_target:6*width_target] = 0
        flattened_context_uint8[5*width_target**2 - 2*width_target:5*width_target**2 - width_target] = 60
        image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                               width_target)
        tls.save_image('sets/pseudo_visualization/arrange_flattened_context.png',
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_build_batch_training(self):
        """Tests the function `build_batch_training` in the file "sets/reading.py".
        
        If `is_pair` is True, for i = 0 ... 3, an image is saved at
        "sets/pseudo_visualization/build_batch_training/width_target_32/pair/image_i.png".
        The test is successful if, in each image, the
        context contains HEVC compression artifacts. However,
        the target patch has no HEVC compression artifact.
        If `is_pair` is False, for i = 0 ... 3, an image is saved at
        "sets/pseudo_visualization/build_batch_training/fully_connected/single/image_i.png".
        The test is successful if, in each image, both
        the context and the target patch have no HEVC
        compression artifact.
        
        """
        is_pair = False
        batch_size = 4
        nb_threads = 2
        
        # Each width of target patch in {4, 8, 16, 32, 64}
        # can be tested.
        width_target = 32
        mean_training = 114.5
        tuple_width_height_masks = ()
        is_fully_connected = False
        
        if is_pair:
            tag_pair = 'pair'
        else:
            tag_pair = 'single'
        paths_to_directory_threads = os.path.join('sets/pseudo_data/build_batch_training/',
                                                  'width_target_{}'.format(width_target),
                                                  tag_pair)
        node_list_batches_float32 = sets.reading.build_batch_training(paths_to_directory_threads,
                                                                      batch_size,
                                                                      nb_threads,
                                                                      width_target,
                                                                      (False,),
                                                                      mean_training,
                                                                      tuple_width_height_masks,
                                                                      is_fully_connected)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            list_batches_float32 = sess.run(node_list_batches_float32)
            coordinator.request_stop()
            coordinator.join(list_threads)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/build_batch_training/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # If `is_fully_connected` is True, `list_batches_float32[0]`
        # is a batch of flattened masked contexts. `list_batches_float32[1]`
        # is a batch of target patches. If `is_fully_connected`
        # is False, `list_batches_float32[0]` is a batch of masked
        # context portions, each being located above a different
        # target patch. `list_batches_float32[1]` is a batch of masked
        # context portions, each being located on the left side of
        # a different target patch. `list_batches_float32[2]` is a
        # batch of target patches.
        for i in range(batch_size):
            if is_fully_connected:
                flattened_context_uint8 = tls.cast_float_to_uint8(list_batches_float32[0][i, :] + mean_training)
                target_uint8 = tls.cast_float_to_uint8(list_batches_float32[1][i, :, :, :] + mean_training)
                image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                                       width_target)
            else:
                portion_above_uint8 = tls.cast_float_to_uint8(list_batches_float32[0][i, :, :, :] + mean_training)
                portion_left_uint8 = tls.cast_float_to_uint8(list_batches_float32[1][i, :, :, :] + mean_training)
                target_uint8 = tls.cast_float_to_uint8(list_batches_float32[2][i, :, :, :] + mean_training)
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
            image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_compute_mean_intensities(self):
        """Tests the function `compute_mean_intensities` in the file "sets/writing.py".
        
        The test is successful if the mean pixels intensity over
        the luminance channel of YCbCr images computed by hand is
        almost equal to that computed by the function.
        
        """
        paths_to_rgbs = [
            'sets/pseudo_data/rgb_bride.jpg',
            'sets/pseudo_data/rgb_cliff.jpg',
            'sets/pseudo_data/rgb_jewelry.jpg',
            'sets/pseudo_data/rgb_web.jpg'
        ]
        
        queue_threads = queue.Queue(maxsize=1)
        sets.writing.compute_mean_intensities(paths_to_rgbs,
                                              queue_threads)
        mean_intensities_float64 = queue_threads.get()
        print('Mean pixels intensity over the luminance channel of YCbCr images computed by hand: {}'.format(95.86575601))
        print('Mean pixels intensity over the luminance channel of YCbCr images computed by `compute_mean_intensities`: {}'.format(mean_intensities_float64[0].item()))
    
    def test_compute_mean_intensities_threading(self):
        """Tests the function `compute_mean_intensities_threading` in the file "sets/writing.py".
        
        The test is successful if the mean pixels intensity over
        the luminance channel of YCbCr images computed by hand is
        almost equal to that computed by the function.
        
        """
        paths_to_rgbs = [
            'sets/pseudo_data/rgb_bride.jpg',
            'sets/pseudo_data/rgb_cliff.jpg',
            'sets/pseudo_data/rgb_jewelry.jpg',
            'sets/pseudo_data/rgb_web.jpg'
        ]
        nb_threads = 2
        
        mean_intensities_float64 = sets.writing.compute_mean_intensities_threading(paths_to_rgbs,
                                                                                   nb_threads)
        print('Mean pixels intensity over the luminance channel of YCbCr images computed by hand: {}'.format(95.86575601))
        print('Mean pixels intensity over the luminance channel of YCbCr images computed by `compute_mean_intensities_threading`: {}'.format(mean_intensities_float64[0].item()))
    
    def test_count_examples_in_files_tfrecord(self):
        """Tests the function `count_examples_in_files_tfrecord` in the file "imagenet/writing.py".
        
        The test is successful if the total number of examples
        in the files ".tfrecord" stored in the subdirectories
        of the directory "sets/pseudo_data/count_examples_in_files_tfrecord/"
        is equal to 5.
        
        """
        path_to_directory_threads = 'sets/pseudo_data/count_examples_in_files_tfrecord/'
        
        nb_examples = sets.writing.count_examples_in_files_tfrecord(path_to_directory_threads)
        print('Total number of examples in the files ".tfrecord" stored in the subdirectories of the directory "{0}": {1}'.format(path_to_directory_threads, nb_examples))
    
    def test_create_example_channel_single_or_pair(self):
        """Tests the function `create_example_channel_single_or_pair` in the file "sets/writing.py".
        
        The test is successful if the data of the
        image channel is bytes.
        
        """
        channel_single_or_pair_uint8 = numpy.ones((4, 4, 2), dtype=numpy.uint8)
        example = sets.writing.create_example_channel_single_or_pair(channel_single_or_pair_uint8)
        print('Data of the image channel:')
        print(example.features.feature['channel_single_or_pair'])
    
    def test_create_example_context_portions_target(self):
        """Tests the function `create_example_context_portions_target` in the file "sets/writing.py".
        
        The test is successful if the data of the
        context portion located above the target patch,
        the data of the context portion located on the
        left side of the target patch and the data of the
        target patch are bytes.
        
        """
        width_target = 8
        
        portion_above_uint8 = numpy.zeros((width_target, 3*width_target, 1),
                                          dtype=numpy.uint8)
        portion_left_uint8 = numpy.ones((2*width_target, width_target, 1),
                                        dtype=numpy.uint8)
        target_uint8 = numpy.ones((width_target, width_target, 1),
                                  dtype=numpy.uint8)
        example = sets.writing.create_example_context_portions_target(portion_above_uint8,
                                                                      portion_left_uint8,
                                                                      target_uint8)
        print('Data of the context portion located above the target patch:')
        print(example.features.feature['portion_above'])
        print('Data of the context portion located on the left side of the target patch:')
        print(example.features.feature['portion_left'])
        print('Data of the target patch:')
        print(example.features.feature['target'])
    
    def test_create_tfrecord(self):
        """Tests the function `create_tfrecord` in the file "sets/writing.py".
        
        If `is_pair` is True, for i = 0 ... 29, an image is saved at
        "sets/pseudo_visualization/create_tfrecord/width_target_32/pair/luminance/image_i.png".
        The test is successful if, in each image, the
        context contains HEVC compression artifacts. However,
        the target patch has no HEVC compression artifact.
        If `is_pair` is False, for i = 0 ... 29, an image is saved at
        "sets/pseudo_visualization/create_tfrecord/width_target_32/single/luminance/image_i.png".
        The test is successful if, in each image, both
        the context and the target patch have no HEVC
        compression artifact.
        Besides, for i = 0 ... 9, the images come from
        a scenery. Similarly, for i = 10 ... 19, the images
        come from another scenery. For i = 20 ... 29, the
        images com from another scenery.
        
        """
        width_target = 32
        is_pair = False
        index_channel = 0
        paths_to_rgbs = [
            'sets/pseudo_data/rgb_library.jpg',
            'sets/pseudo_data/rgb_web.jpg',
            'sets/pseudo_data/rgb_cliff.jpg'
        ]
        path_to_directory_temp = 'sets/pseudo_data/temp/create_tfrecord/'
        if is_pair:
            dict_pair = {
                'path_to_before_encoding_hevc': os.path.join(path_to_directory_temp,
                                                             'image_before_encoding_hevc.yuv'),
                'path_to_after_encoding_hevc': os.path.join(path_to_directory_temp,
                                                            'image_after_encoding_hevc.yuv'),
                'path_to_cfg': 'hevc/configuration/intra_main.cfg',
                'path_to_bitstream': os.path.join(path_to_directory_temp,
                                                  'bitstream.bin'),
                'path_to_exe_encoder': hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                'qps_int': numpy.array([32, 42],
                                       dtype=numpy.int32)
            }
            tag_pair = 'pair'
        else:
            dict_pair = None
            tag_pair = 'single'
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        
        path_to_directory_tfrecord = os.path.join('sets/pseudo_data/create_tfrecord/',
                                                  'width_target_{}'.format(width_target),
                                                  tag_pair,
                                                  tag_channel)
        
        # The directory containing the file ".tfrecord" is
        # created if it does not exist.
        if not os.path.isdir(path_to_directory_tfrecord):
            os.makedirs(path_to_directory_tfrecord)
        path_to_tfrecord = os.path.join(path_to_directory_tfrecord,
                                        'data.tfrecord')
        sets.writing.create_tfrecord(path_to_tfrecord,
                                     paths_to_rgbs,
                                     width_target,
                                     index_channel,
                                     dict_pair)
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.read_queue_tfrecord(queue_tfrecord,
                                             width_target,
                                             (False,))
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/create_tfrecord/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            
            # For each RGB image that is not dumped, 10 target patches,
            # each paired with its two context portions, are written to
            # the file ".tfrecord".
            for i in range(10*len(paths_to_rgbs)):
                [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                    [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
                )
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_create_tfrecord_batch(self):
        """Tests the function `create_tfrecord_batch` in the file "sets/writing.py".
        
        If `is_pair` is True, for i = 0 ... 29, an image is saved at
        "sets/pseudo_visualization/create_tfrecord_batch/width_target_32/pair/luminance/image_i.png".
        The test is successful if, in each image, the
        context contains HEVC compression artifacts. However,
        the target patch has no HEVC compression artifact.
        If `is_pair` is False, for i = 0 ... 29, an image is saved at
        "sets/pseudo_visualization/create_tfrecord_batch/width_target_32/single/luminance/image_i.png".
        The test is successful if, in each image, both
        the context and the target patch have no HEVC
        compression artifact.
        Besides, for i = 0 ... 9, the images come from
        a scenery. Similarly, for i = 10 ... 19, the images
        come from another scenery. For i = 20 ... 29, the
        images com from another scenery.
        
        """
        width_target = 32
        is_pair = False
        index_channel = 0
        paths_to_rgbs = [
            'sets/pseudo_data/rgb_library.jpg',
            'sets/pseudo_data/rgb_web.jpg',
            'sets/pseudo_data/rgb_cliff.jpg'
        ]
        nb_rgbs_per_tfrecord = 2
        path_to_directory_temp = 'sets/pseudo_data/temp/create_tfrecord_batch/'
        if is_pair:
            dict_pair = {
                'path_to_before_encoding_hevc': os.path.join(path_to_directory_temp,
                                                             'image_before_encoding_hevc.yuv'),
                'path_to_after_encoding_hevc': os.path.join(path_to_directory_temp,
                                                            'image_after_encoding_hevc.yuv'),
                'path_to_cfg': 'hevc/configuration/intra_main.cfg',
                'path_to_bitstream': os.path.join(path_to_directory_temp,
                                                  'bitstream.bin'),
                'path_to_exe_encoder': hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                'qps_int': numpy.array([32, 42],
                                       dtype=numpy.int32)
            }
            tag_pair = 'pair'
        else:
            dict_pair = None
            tag_pair = 'single'
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        
        path_to_directory_tfrecords = os.path.join('sets/pseudo_data/create_tfrecord_batch/',
                                                   'width_target_{}'.format(width_target),
                                                   tag_pair,
                                                   tag_channel)
        
        # The directory containing the files ".tfrecord" is
        # created if it does not exist.
        if not os.path.isdir(path_to_directory_tfrecords):
            os.makedirs(path_to_directory_tfrecords)
        sets.writing.create_tfrecord_batch(path_to_directory_tfrecords,
                                           paths_to_rgbs,
                                           width_target,
                                           index_channel,
                                           nb_rgbs_per_tfrecord,
                                           dict_pair)
        
        # The files ".tfrecord" the directory at `path_to_directory_tfrecords`
        # contains are collected.
        paths_to_tfrecords = []
        for name_item in os.listdir(path_to_directory_tfrecords):
            if name_item.endswith('.tfrecord'):
                paths_to_tfrecords.append(os.path.join(path_to_directory_tfrecords, name_item))
        queue_tfrecord = tf.train.string_input_producer(paths_to_tfrecords,
                                                        shuffle=False)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.read_queue_tfrecord(queue_tfrecord,
                                             width_target,
                                             (False,))
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/create_tfrecord_batch/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            for i in range(10*len(paths_to_rgbs)):
                [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                    [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
                )
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_create_tfrecord_threading(self):
        """Tests the function `create_tfrecord_threading` in the file "sets/writing.py".
        
        If `is_pair` is True, for i = 0 ... 49, an image is saved at
        "sets/pseudo_visualization/create_tfrecord_threading/width_target_32/pair/luminance/image_i.png".
        The test is successful if, in each image, the
        context contains HEVC compression artifacts. However,
        the target patch has no HEVC compression artifact.
        If `is_pair` is False, for i = 0 ... 49, an image is saved at
        "sets/pseudo_visualization/create_tfrecord_threading/width_target_32/single/luminance/image_i.png".
        The test is successful if, in each image, both
        the context and the target patch have no HEVC
        compression artifact.
        Besides, for i = 0 ... 9, the images come from
        a scenery. Similarly, for i = 10 ... 19, the images
        come from another scenery. For i = 20 ... 29, the
        images com from another scenery. And so on ...
        
        """
        width_target = 32
        is_pair = False
        index_channel = 0
        paths_to_rgbs = [
            'sets/pseudo_data/cmyk_snake.jpg',
            'sets/pseudo_data/rgb_bride.jpg',
            'sets/pseudo_data/rgb_cliff.jpg',
            'sets/pseudo_data/rgb_jewelry.jpg',
            'sets/pseudo_data/rgb_library.jpg',
            'sets/pseudo_data/rgb_web.jpg'
        ]
        nb_rgbs_per_tfrecord = 2
        nb_threads = 2
        path_to_directory_temp = 'sets/pseudo_data/temp/create_tfrecord_threading/'
        if is_pair:
            dict_pair_threads = {
                'paths_to_before_encoding_hevc': [
                    os.path.join(path_to_directory_temp, 'image_before_encoding_hevc_{}.yuv'.format(i)) for i in range(nb_threads)
                ],
                'paths_to_after_encoding_hevc': [
                    os.path.join(path_to_directory_temp, 'image_after_encoding_hevc_{}.yuv'.format(i)) for i in range(nb_threads)
                ],
                'path_to_cfg': 'hevc/configuration/intra_main.cfg',
                'paths_to_bitstream': [
                    os.path.join(path_to_directory_temp, 'bitstream_{}.bin'.format(i)) for i in range(nb_threads)
                ],
                'path_to_exe_encoder': hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                'qps_int': numpy.array([32, 42],
                                       dtype=numpy.int32)
            }
            tag_pair = 'pair'
        else:
            dict_pair_threads = None
            tag_pair = 'single'
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        
        # The paths to the subdirectories of the directory
        # at `path_to_directory_threads` are in the list
        # `paths_to_directories_tfrecords`.
        path_to_directory_threads = os.path.join('sets/pseudo_data/create_tfrecord_threading/',
                                                 'width_target_{}'.format(width_target),
                                                 tag_pair,
                                                 tag_channel)
        if not os.path.isdir(path_to_directory_threads):
            os.makedirs(path_to_directory_threads)
        paths_to_directories_tfrecords = []
        for i in range(nb_threads):
            path_to_directory_tfrecord = os.path.join(path_to_directory_threads,
                                                      'thread_{}'.format(i))
            if not os.path.isdir(path_to_directory_tfrecord):
                os.mkdir(path_to_directory_tfrecord)
            paths_to_directories_tfrecords.append(path_to_directory_tfrecord)
        sets.writing.create_tfrecord_threading(paths_to_directories_tfrecords,
                                               paths_to_rgbs,
                                               width_target,
                                               index_channel,
                                               nb_rgbs_per_tfrecord,
                                               dict_pair_threads)
        paths_to_tfrecords = tls.collect_paths_to_files_in_subdirectories(path_to_directory_threads,
                                                                          '.tfrecord')
        queue_tfrecord = tf.train.string_input_producer(paths_to_tfrecords,
                                                        shuffle=False)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.read_queue_tfrecord(queue_tfrecord,
                                             width_target,
                                             (False,))
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/create_tfrecord_threading/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            
            # `tf.train.start_queue_runners` starts all queue
            # runners in the collection of key "queue_runners".
            # `tf.train.start_queue_runners` returns a list of
            # threads.
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            
            # One image out of 6 is skipped as the mode of
            # the image is not RGB.
            for i in range(10*(len(paths_to_rgbs) - 1)):
                [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                    [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
                )
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_create_training_subset(self):
        """Tests the function `create_training_subset` in the file "sets/writing.py".
        
        The test is successful if the directory at
        "sets/pseudo_visualization/create_training_subset/luminance/"
        contains the same channel from 3 different
        YCbCr images.
        
        """
        paths_to_rgbs = [
            'sets/pseudo_data/cmyk_snake.jpg',
            'sets/pseudo_data/rgb_bride.jpg',
            'sets/pseudo_data/rgb_cliff.jpg',
            'sets/pseudo_data/rgb_jewelry.jpg',
            'sets/pseudo_data/rgb_library.jpg',
            'sets/pseudo_data/rgb_web.jpg'
        ]
        path_to_training_subset = 'sets/pseudo_data/create_training_subset/training_subset.npy'
        nb_examples = 3
        
        sets.writing.create_training_subset(paths_to_rgbs,
                                            path_to_training_subset,
                                            nb_examples)
        training_subset_uint8 = numpy.load(path_to_training_subset)
        path_to_directory_vis = 'sets/pseudo_visualization/create_training_subset/'
        for i in range(nb_examples):
            tls.save_image(os.path.join(path_to_directory_vis, 'luminance_{}.png'.format(i)),
                           training_subset_uint8[i, :, :, 0])
            tls.save_image(os.path.join(path_to_directory_vis, 'chrominance_blue_{}.png'.format(i)),
                           training_subset_uint8[i, :, :, 1])
            tls.save_image(os.path.join(path_to_directory_vis, 'chrominance_red_{}.png'.format(i)),
                           training_subset_uint8[i, :, :, 2])
    
    def test_extract_context_portions_target_from_channel_numpy(self):
        """Tests the function `extract_context_portions_target_from_channel_numpy` in the file "sets/common.py".
        
        If `is_pair` is True, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_target_from_channel_numpy/width_target_64/pair/image.png".
        The test is successful if, in this image, the
        context contains uniform noise. However, the
        target patch has no uniform noise.
        If `is_pair` is False, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_target_from_channel_numpy/width_target_64/single/image.png".
        The test is successful if both the context
        and the target patch have no uniform noise.
        
        """
        width_target = 64
        is_pair = False
        row_1st = 449
        col_1st = 768
        index_channel = 0
        
        rgb_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                        'RGB')
        channel_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, index_channel:index_channel + 1]
        
        # If `is_pair` is True, an image channel and this image
        # channel with HEVC compression artifacts are inserted into
        # `sets.common.extract_context_portions_target_from_channel_numpy`.
        # If `is_pair` is False, only this image channel is inserted into
        # `sets.common.extract_context_portions_target_from_channel_numpy`.
        if is_pair:
            tag_pair = 'pair'
            
            # The HEVC compression artifacts are viewed as
            # uniform noise for simplicity.
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channel_uint8.shape)
            channel_artifacts_uint8 = tls.cast_float_to_uint8(channel_uint8.astype(numpy.float64) + noise_uniform)
            channel_single_or_pair_uint8 = numpy.concatenate((channel_uint8, channel_artifacts_uint8),
                                                             axis=2)
        else:
            tag_pair = 'single'
            channel_single_or_pair_uint8 = channel_uint8
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        (portion_above_uint8, portion_left_uint8, target_uint8) = \
            sets.common.extract_context_portions_target_from_channel_numpy(channel_single_or_pair_uint8,
                                                                           width_target,
                                                                           row_1st,
                                                                           col_1st)
        image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                              portion_left_uint8)
        image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/extract_context_portions_target_from_channel_numpy/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        
        # The directory containing the saved image is created
        # if it does not exist.
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        tls.save_image(os.path.join(path_to_directory_vis, 'image.png'),
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_context_portions_target_from_channel_tf(self):
        """Tests the function `extract_context_portions_target_from_channel_tf` in the file "sets/reading.py".
        
        If `is_pair` is True, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_target_from_channel_tf/width_target_32/pair/image.png".
        If `is_pair` is False, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_target_from_channel_tf/width_target_32/single/image.png".
        The test is successful if this image is different
        every time the test is re-run. It is due to the
        random crop.
        
        """
        width_target = 32
        is_pair = False
        if is_pair:
            tag_pair = 'pair'
        else:
            tag_pair = 'single'
        path_to_tfrecord = os.path.join('sets/pseudo_data/extract_context_portions_target_from_channel_tf/',
                                        tag_pair,
                                        'data.tfrecord')
        
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        reader = tf.TFRecordReader()
        serialized_example = reader.read(queue_tfrecord)[1]
        node_channel_single_or_pair_uint8 = sets.reading.parse_example_channel_single_or_pair(serialized_example,
                                                                                              is_pair)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.extract_context_portions_target_from_channel_tf(node_channel_single_or_pair_uint8,
                                                                         width_target)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
            )
            coordinator.request_stop()
            coordinator.join(list_threads)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/extract_context_portions_target_from_channel_tf/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        
        # The directory containing the saved image is created
        # if it does not exist.
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                              portion_left_uint8)
        image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
        tls.save_image(os.path.join(path_to_directory_vis, 'image.png'),
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_context_portions_targets_from_channel_numpy(self):
        """Tests the function `extract_context_portions_targets_from_channel_numpy` in the file "sets/common.py".
        
        If `is_pair` is True, for i = 0 ... 3, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_targets_from_channel_numpy/width_target_64/pair/image_i.png".
        The test is successful if, in each image, the context
        contains uniform noise. However, the target patch has
        no uniform noise.
        If `is_pair` is False, for i = 0 ... 3, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_targets_from_channel_numpy/width_target_64/single/image_i.png".
        The test is successful if, in each image, both the
        context and the target patch have no uniform noise.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([449, 449, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([768, 0, 768, 0], dtype=numpy.uint32)
        index_channel = 0
        
        rgb_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                        'RGB')
        channel_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, index_channel:index_channel + 1]
        if is_pair:
            tag_pair = 'pair'
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channel_uint8.shape)
            channel_artifacts_uint8 = tls.cast_float_to_uint8(channel_uint8.astype(numpy.float64) + noise_uniform)
            channel_single_or_pair_uint8 = numpy.concatenate((channel_uint8, channel_artifacts_uint8),
                                                             axis=2)
        else:
            tag_pair = 'single'
            channel_single_or_pair_uint8 = channel_uint8
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        (portions_above_uint8, portions_left_uint8, targets_uint8) = \
            sets.common.extract_context_portions_targets_from_channel_numpy(channel_single_or_pair_uint8,
                                                                            width_target,
                                                                            row_1sts,
                                                                            col_1sts)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/extract_context_portions_targets_from_channel_numpy/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # `portions_above_uint8.shape[0]`, `portions_left_uint8.shape[0]`
        # and `targets_uint8.shape[0]` are equal to `row_1sts.size`.
        for i in range(portions_above_uint8.shape[0]):
            image_uint8 = sets.arranging.arrange_context_portions(portions_above_uint8[i, :, :, :],
                                                                  portions_left_uint8[i, :, :, :])
            image_uint8[width_target:2*width_target, width_target:2*width_target, :] = targets_uint8[i, :, :, :]
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_context_portions_targets_from_channels_numpy(self):
        """Tests the function `extract_context_portions_targets_from_channels_numpy` in the file "sets/common.py".
        
        If `is_pair` is True, for i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_targets_from_channels_numpy/width_target_64/pair/image_i.png".
        The test is successful if, in each image, the context
        contains uniform noise. However, the target patch has
        no uniform noise.
        If `is_pair` is False, for i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_targets_from_channels_numpy/width_target_64/single/image_i.png".
        The test is successful if, in each image, both the
        context and the target patch have no uniform noise.
        Besides, whatever the value of `is_pair`, for i = 0 ... 3,
        the contexts and the target patches all come
        from an image channel. For i = 4 ... 7, the
        contexts and the target patches all come from
        the same channel of another image.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([449, 449, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([768, 0, 768, 0], dtype=numpy.uint32)
        index_channel = 0
        
        rgb_0_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        channel_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[:, :, index_channel:index_channel + 1]
        rgb_1_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_library.jpg',
                                          'RGB')
        channel_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[:, :, index_channel:index_channel + 1]
        channels_uint8 = numpy.stack((channel_0_uint8, channel_1_uint8),
                                     axis=0)
        if is_pair:
            tag_pair = 'pair'
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channels_uint8.shape)
            channels_artifacts_uint8 = tls.cast_float_to_uint8(channels_uint8.astype(numpy.float64) + noise_uniform)
            channels_single_or_pair_uint8 = numpy.concatenate((channels_uint8, channels_artifacts_uint8),
                                                              axis=3)
        else:
            tag_pair = 'single'
            channels_single_or_pair_uint8 = channels_uint8
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        (portions_above_uint8, portions_left_uint8, targets_uint8) = \
            sets.common.extract_context_portions_targets_from_channels_numpy(channels_single_or_pair_uint8,
                                                                             width_target,
                                                                             row_1sts,
                                                                             col_1sts)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/extract_context_portions_targets_from_channels_numpy/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # `portions_above_uint8.shape[0]`, `portions_left_uint8.shape[0]`
        # and `targets_uint8.shape[0]` are equal to `2*row_1sts.size`.
        for i in range(portions_above_uint8.shape[0]):
            image_uint8 = sets.arranging.arrange_context_portions(portions_above_uint8[i, :, :, :],
                                                                  portions_left_uint8[i, :, :, :])
            image_uint8[width_target:2*width_target, width_target:2*width_target, :] = targets_uint8[i, :, :, :]
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_context_portions_targets_from_channels_plus_preprocessing(self):
        """Tests the function `extract_context_portions_targets_from_channels_plus_preprocessing` in the file "sets/common.py".
        
        If `is_pair` is True, for i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_targets_from_channels_plus_preprocessing/width_target_64/pair/image_i.png".
        If `is_pair` is False, for i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/extract_context_portions_targets_from_channels_plus_preprocessing/width_target_64/single/image_i.png".
        The test is successful if, in all images, we
        retrieve the same two masks (in gray).
        
        """
        width_target = 64
        is_pair = False
        is_fully_connected = False
        row_1sts = numpy.array([449, 449, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([768, 0, 768, 0], dtype=numpy.uint32)
        index_channel = 0
        
        # `mean_training` is a fake mean pixels intensity here.
        mean_training = 114.5
        tuple_width_height_masks = (4, 32)
        
        rgb_0_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        channel_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[:, :, index_channel:index_channel + 1]
        rgb_1_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_library.jpg',
                                          'RGB')
        channel_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[:, :, index_channel:index_channel + 1]
        channels_uint8 = numpy.stack((channel_0_uint8, channel_1_uint8),
                                     axis=0)
        if is_pair:
            tag_pair = 'pair'
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channels_uint8.shape)
            channels_artifacts_uint8 = tls.cast_float_to_uint8(channels_uint8.astype(numpy.float64) + noise_uniform)
            channels_single_or_pair_uint8 = numpy.concatenate((channels_uint8, channels_artifacts_uint8),
                                                              axis=3)
        else:
            tag_pair = 'single'
            channels_single_or_pair_uint8 = channels_uint8
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        tuple_batches_float32 = sets.common.extract_context_portions_targets_from_channels_plus_preprocessing(channels_single_or_pair_uint8,
                                                                                                              width_target,
                                                                                                              row_1sts,
                                                                                                              col_1sts,
                                                                                                              mean_training,
                                                                                                              tuple_width_height_masks,
                                                                                                              is_fully_connected)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/extract_context_portions_targets_from_channels_plus_preprocessing/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        for i in range(tuple_batches_float32[0].shape[0]):
            if is_fully_connected:
                
                # `tls.cast_float_to_uint8` systematically replaces
                # the standard cast from float to 8-bit unsigned integer
                # via `numpy.ndarray.astype`.
                flattened_context_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[0][i, :] + mean_training)
                target_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[1][i, :, :, :] + mean_training)
                image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                                       width_target)
            else:
                portion_above_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[0][i, :, :, :] + mean_training)
                portion_left_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[1][i, :, :, :] + mean_training)
                target_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[2][i, :, :, :] + mean_training)
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
            image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_context_portions_targets_plus_writing(self):
        """Tests the function `extract_context_portions_targets_plus_writing` in the file "sets/writing.py".
        
        The condition of success is the same as that of
        `test_extract_context_portions_targets_from_channel_numpy`.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([449, 449, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([768, 0, 768, 0], dtype=numpy.uint32)
        path_to_tfrecord = 'sets/pseudo_data/extract_context_portions_targets_plus_writing/data.tfrecord'
        
        rgb_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                        'RGB')
        channel_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, 0:1]
        if is_pair:
            tag_pair = 'pair'
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channel_uint8.shape)
            channel_artifacts_uint8 = tls.cast_float_to_uint8(channel_uint8.astype(numpy.float64) + noise_uniform)
            channel_single_or_pair_uint8 = numpy.concatenate((channel_uint8, channel_artifacts_uint8),
                                                             axis=2)
        else:
            tag_pair = 'single'
            channel_single_or_pair_uint8 = channel_uint8
        with tf.python_io.TFRecordWriter(path_to_tfrecord) as file:
            sets.writing.extract_context_portions_targets_plus_writing(channel_single_or_pair_uint8,
                                                                       width_target,
                                                                       row_1sts,
                                                                       col_1sts,
                                                                       file)
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.read_queue_tfrecord(queue_tfrecord,
                                             width_target,
                                             (False,))
        
        # For two different widths of target patch, the visualizations
        # are stored in two different directories.
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/extract_context_portions_targets_plus_writing/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            for i in range(2):
                [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                    [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
                )
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_parse_example_channel_single_or_pair(self):
        """Tests the function `parse_example_channel_single_or_pair` in the file "sets/reading.py".
        
        If `is_pair` is True, a 1st image is saved at
        "sets/pseudo_visualization/parse_example_channel_single_or_pair/pair/channel_original.png"
        and a 2nd image is saved at
        "sets/pseudo_visualization/parse_example_channel_single_or_pair/pair/channel_artifacts.png".
        The test is successful if the 1st image has
        no uniform noise whereas the 2nd image contains
        uniform noise.
        If `is_pair` is False, an image is saved at
        "sets/pseudo_visualization/parse_example_channel_single_or_pair/single/channel.png".
        The test is successful if the image has no
        uniform noise.
        
        """
        is_pair = False
        if is_pair:
            tag_pair = 'pair'
        else:
            tag_pair = 'single'
        path_to_tfrecord = os.path.join('sets/pseudo_data/parse_example_channel_single_or_pair/',
                                        tag_pair,
                                        'data.tfrecord')
        
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        reader = tf.TFRecordReader()
        serialized_example = reader.read(queue_tfrecord)[1]
        node_channel_single_or_pair_uint8 = sets.reading.parse_example_channel_single_or_pair(serialized_example,
                                                                                              is_pair)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            channel_out_uint8 = sess.run(node_channel_single_or_pair_uint8)
            coordinator.request_stop()
            coordinator.join(list_threads)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/parse_example_channel_single_or_pair/',
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        if is_pair:
            tls.save_image(os.path.join(path_to_directory_vis, 'channel_original.png'),
                           channel_out_uint8[:, :, 0])
            tls.save_image(os.path.join(path_to_directory_vis, 'channel_artifacts.png'),
                           channel_out_uint8[:, :, 1])
        else:
            tls.save_image(os.path.join(path_to_directory_vis, 'channel.png'),
                           numpy.squeeze(channel_out_uint8, axis=2))
    
    def test_parse_example_context_portions_target(self):
        """Tests the function `parse_example_context_portions_target` in the file "sets/reading.py".
        
        An image is saved at
        "sets/pseudo_visualization/parse_example_context_portions_target.png".
        The test is successful if there is no
        discontinuity between the target patch
        and its context in this image.
        
        """
        path_to_tfrecord = 'sets/pseudo_data/parse_example_context_portions_target/data.tfrecord'
        
        # We know the width of the target
        # patch that was used when the file
        # at "sets/pseudo_data/parse_example_context_portions_target/data.tfrecord"
        # was created. `width_target` cannot
        # be changed.
        width_target = 64
        
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        reader = tf.TFRecordReader()
        serialized_example = reader.read(queue_tfrecord)[1]
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.parse_example_context_portions_target(serialized_example,
                                                               width_target)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
            )
            coordinator.request_stop()
            coordinator.join(list_threads)
        image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                              portion_left_uint8)
        image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
        tls.save_image('sets/pseudo_visualization/parse_example_context_portions_target.png',
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_preprocess_context_portions_target_tf(self):
        """Tests the function `preprocess_context_portions_target_tf` in the file "sets/reading.py".
        
        For i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/preprocess_context_portions_target_tf/width_target_32/image_i.png".
        The test is successful if, for each image,
        its aspect is unchanged whether `is_fully_connected`
        is True or False.
        
        """
        # Each width of target patch in {4, 8, 16, 32, 64}
        # can be tested.
        width_target = 32
        mean_training = 114.5
        tuple_width_height_masks = ()
        is_fully_connected = False
        path_to_tfrecord = os.path.join('sets/pseudo_data/preprocess_context_portions_target_tf/',
                                        'width_target_{}'.format(width_target),
                                        'data.tfrecord')
        
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = sets.reading.read_queue_tfrecord(queue_tfrecord,
                                                                                                                  width_target,
                                                                                                                  (False,))
        node_tuple_context_portions_target_float32 = \
            sets.reading.preprocess_context_portions_target_tf(node_portion_above_uint8,
                                                               node_portion_left_uint8,
                                                               node_target_uint8,
                                                               mean_training,
                                                               tuple_width_height_masks,
                                                               is_fully_connected)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/preprocess_context_portions_target_tf/',
                                             'width_target_{}'.format(width_target))
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            for i in range(8):
                tuple_context_portions_target_float32 = sess.run(node_tuple_context_portions_target_float32)
                if is_fully_connected:
                    flattened_context_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[0] + mean_training)
                    target_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[1] + mean_training)
                    image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                                           width_target)
                else:
                    portion_above_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[0] + mean_training)
                    portion_left_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[1] + mean_training)
                    target_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[2] + mean_training)
                    image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                          portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_preprocess_context_portions_targets_numpy(self):
        """Tests the function `preprocess_context_portions_targets_numpy` in the file "sets/common.py".
        
        If `is_pair` is True, for i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/preprocess_context_portions_targets_numpy/width_target_64/pair/image_i.png".
        If `is_pair` is False, for i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/preprocess_context_portions_targets_numpy/width_target_64/single/image_i.png".
        The test is successful if, in all images, we
        retrieve the same two masks (in gray).
        
        """
        width_target = 64
        is_pair = False
        is_fully_connected = False
        row_1sts = numpy.array([449, 449, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([768, 0, 768, 0], dtype=numpy.uint32)
        index_channel = 0
        
        # `mean_training` is a fake mean pixels intensity here.
        mean_training = 114.5
        tuple_width_height_masks = (32, 16)
        
        rgb_0_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        channel_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[:, :, index_channel:index_channel + 1]
        rgb_1_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_library.jpg',
                                          'RGB')
        channel_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[:, :, index_channel:index_channel + 1]
        channels_uint8 = numpy.stack((channel_0_uint8, channel_1_uint8),
                                     axis=0)
        if is_pair:
            tag_pair = 'pair'
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channels_uint8.shape)
            channels_artifacts_uint8 = tls.cast_float_to_uint8(channels_uint8.astype(numpy.float64) + noise_uniform)
            channels_single_or_pair_uint8 = numpy.concatenate((channels_uint8, channels_artifacts_uint8),
                                                              axis=3)
        else:
            tag_pair = 'single'
            channels_single_or_pair_uint8 = channels_uint8
        if index_channel == 0:
            tag_channel = 'luminance'
        elif index_channel == 1:
            tag_channel = 'chrominance_blue'
        else:
            tag_channel = 'chrominance_red'
        (portions_above_uint8, portions_left_uint8, targets_uint8) = \
            sets.common.extract_context_portions_targets_from_channels_numpy(channels_single_or_pair_uint8,
                                                                             width_target,
                                                                             row_1sts,
                                                                             col_1sts)
        tuple_batches_float32 = sets.common.preprocess_context_portions_targets_numpy(portions_above_uint8,
                                                                                      portions_left_uint8,
                                                                                      targets_uint8,
                                                                                      mean_training,
                                                                                      tuple_width_height_masks,
                                                                                      is_fully_connected)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/preprocess_context_portions_targets_numpy/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair,
                                             tag_channel)
        
        # The directory containing the saved image is created
        # if it does not exist.
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        for i in range(tuple_batches_float32[0].shape[0]):
            if is_fully_connected:
                flattened_context_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[0][i, :] + mean_training)
                target_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[1][i, :, :, :] + mean_training)
                image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                                       width_target)
            else:
                portion_above_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[0][i, :, :, :] + mean_training)
                portion_left_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[1][i, :, :, :] + mean_training)
                target_uint8 = tls.cast_float_to_uint8(tuple_batches_float32[2][i, :, :, :] + mean_training)
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
            image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_read_queue_tfrecord(self):
        """Tests the function `read_queue_tfrecord` in the file "sets/reading.py".
        
        If `is_pair` is True, an image is saved at
        "sets/pseudo_visualization/read_queue_tfrecord/width_target_16/pair/image.png".
        The test is successful if, in this image, the
        context contains HEVC compression artifacts. However,
        the target patch has no HEVC compression artifact.
        If `is_pair` is False, an image is saved at
        "sets/pseudo_visualization/read_queue_tfrecord/width_target_16/single/image.png".
        The test is successful if, in this image, both
        the context and the target patch have no HEVC
        compression artifact.
        
        """
        # Each width of target patch in {4, 8, 16, 32, 64}
        # can be tested.
        width_target = 16
        is_pair = False
        if is_pair:
            tag_pair = 'pair'
        else:
            tag_pair = 'single'
        path_to_tfrecord = os.path.join('sets/pseudo_data/read_queue_tfrecord/',
                                        'width_target_{}'.format(width_target),
                                        tag_pair,
                                        'data.tfrecord')
        
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.read_queue_tfrecord(queue_tfrecord,
                                             width_target,
                                             (False,))
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
            )
            coordinator.request_stop()
            coordinator.join(list_threads)
        image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                              portion_left_uint8)
        image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/read_queue_tfrecord/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        tls.save_image(os.path.join(path_to_directory_vis, 'image.png'),
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_read_queue_tfrecord_plus_preprocessing(self):
        """Tests the function `read_queue_tfrecord_plus_preprocessing` in the file "sets/reading.py".
        
        For i = 0 ... 7, an image is saved at
        "sets/pseudo_visualization/read_queue_tfrecord_plus_preprocessing/width_target_32/image_i.png".
        The test is successful if, for each image,
        its aspect is unchanged whether `is_fully_connected`
        is True or False.
        
        """
        width_target = 32
        mean_training = 114.5
        tuple_width_height_masks = ()
        is_fully_connected = False
        
        # The test of `preprocess_context_portions_target_tf` and that
        # of `read_queue_tfrecord_plus_preprocessing` share the same
        # pseudo data.
        path_to_tfrecord = os.path.join('sets/pseudo_data/preprocess_context_portions_target_tf/',
                                        'width_target_{}'.format(width_target),
                                        'data.tfrecord')
        
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        node_tuple_context_portions_target_float32 = \
            sets.reading.read_queue_tfrecord_plus_preprocessing(queue_tfrecord,
                                                                width_target,
                                                                (False,),
                                                                mean_training,
                                                                tuple_width_height_masks,
                                                                is_fully_connected)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/read_queue_tfrecord_plus_preprocessing/',
                                             'width_target_{}'.format(width_target))
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            for i in range(8):
                tuple_context_portions_target_float32 = sess.run(node_tuple_context_portions_target_float32)
                if is_fully_connected:
                    flattened_context_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[0] + mean_training)
                    target_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[1] + mean_training)
                    image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                                           width_target)
                else:
                    portion_above_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[0] + mean_training)
                    portion_left_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[1] + mean_training)
                    target_uint8 = tls.cast_float_to_uint8(tuple_context_portions_target_float32[2] + mean_training)
                    image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                          portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_rotate_randomly_0_90_180_270(self):
        """Tests the function `rotate_randomly_0_90_180_270` in the file "sets/reading.py".
        
        The test is successful if the channels of the output
        are rotated by the same angle.
        
        """
        input_0_int32 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=numpy.int32)
        input_1_int32 = numpy.array([[101, 102, 103, 104], [105, 106, 107, 108], [109, 110, 111, 112], [113, 114, 115, 116]], dtype=numpy.int32)
        input = numpy.stack((input_0_int32, input_1_int32),
                            axis=2)
        node_input = tf.placeholder(tf.uint8, shape=(4, 4, 2))
        node_output = sets.reading.rotate_randomly_0_90_180_270(node_input)
        with tf.Session() as sess:
            output = sess.run(node_output, feed_dict={node_input:input})
        print('1st channel of the input:')
        print(input[:, :, 0])
        print('2nd channel of the input:')
        print(input[:, :, 1])
        print('1st channel of the output:')
        print(output[:, :, 0])
        print('2nd channel of the output:')
        print(output[:, :, 1])
    
    def test_untar_archive(self):
        """Tests the function `untar_archive` in the file "sets/untar.py".
        
        The test is successful if the directory
        "sets/pseudo_visualization/untar_archive/"
        contains "rgb_tree.jpg" and "rgb_artificial.png".
        
        """
        path_to_directory_extraction = 'sets/pseudo_visualization/untar_archive/'
        path_to_tar = 'sets/pseudo_data/untar_archive/pseudo_archive.tar'
        
        sets.untar.untar_archive(path_to_directory_extraction,
                                 path_to_tar)
    
    def test_untar_ilsvrc2012_training(self):
        """Tests the function `untar_ilsvrc2012_training` in the file "sets/untar.py".
        
        The test is successful if the directory at
        "sets/pseudo_visualization/untar_ilsvrc2012_training/"
        contains 4 images at respectively "directory_0/rgb_bride.jpg",
        "directory_0/rgb_cliff.jpg", "directory_1/rgb_jewelry.jpg",
        "directory_1/rgb_library.jpg".
        
        """
        path_to_directory_extraction = 'sets/pseudo_visualization/untar_ilsvrc2012_training/'
        path_to_tar_ilsvrc2012 = 'sets/pseudo_data/untar_ilsvrc2012_training/pseudo_nested_archive.tar'
        path_to_synsets = 'sets/pseudo_data/untar_ilsvrc2012_training/pseudo_synsets.txt'
        
        sets.untar.untar_ilsvrc2012_training(path_to_directory_extraction,
                                             path_to_tar_ilsvrc2012,
                                             path_to_synsets)
    
    def test_write_channel_single_or_pair(self):
        """Tests the function `write_channel_single_or_pair` in the file "sets/writing.py".
        
        If `is_pair` is True, a 1st image is saved at
        "sets/pseudo_visualization/write_channel_single_or_pair/pair/channel_original.png"
        and a 2nd image is saved at
        "sets/pseudo_visualization/write_channel_single_or_pair/pair/channel_artifacts.png".
        The test is successful if the 1st image has
        no uniform noise whereas the 2nd image contains
        uniform noise.
        If `is_pair` is False, an image is saved at
        "sets/pseudo_visualization/write_channel_single_or_pair/single/channel.png".
        The test is successful if the image has no
        uniform noise.
        
        """
        is_pair = False
        
        rgb_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                        'RGB')
        
        # The height of `rgb_uint8` is equal to 641. Its
        # width is equal to 960. `sets.reading.parse_example_channel_single_or_pair`
        # parses tensors of height and width equal to
        # `sets.writing.WIDTH_CROP` from a file ".tfrecord".
        # The graph must know at construction time the
        # shape of the tensors in a file ".tfrecord".
        channel_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[0:sets.writing.WIDTH_CROP, 0:sets.writing.WIDTH_CROP, 0:1]
        if is_pair:
            tag_pair = 'pair'
            noise_uniform = numpy.random.uniform(low=-50.,
                                                 high=50.,
                                                 size=channel_uint8.shape)
            channel_artifacts_uint8 = tls.cast_float_to_uint8(channel_uint8.astype(numpy.float64) + noise_uniform)
            channel_single_or_pair_uint8 = numpy.concatenate((channel_uint8, channel_artifacts_uint8),
                                                             axis=2)
        else:
            tag_pair = 'single'
            channel_single_or_pair_uint8 = channel_uint8
        path_to_directory_tfrecord = os.path.join('sets/pseudo_data/write_channel_single_or_pair/',
                                                  tag_pair)
        if not os.path.isdir(path_to_directory_tfrecord):
            os.makedirs(path_to_directory_tfrecord)
        path_to_tfrecord = os.path.join(path_to_directory_tfrecord,
                                        'data.tfrecord')
        with tf.python_io.TFRecordWriter(path_to_tfrecord) as file:
            sets.writing.write_channel_single_or_pair(channel_single_or_pair_uint8,
                                                      file)
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        reader = tf.TFRecordReader()
        serialized_example = reader.read(queue_tfrecord)[1]
        node_channel_single_or_pair_uint8 = sets.reading.parse_example_channel_single_or_pair(serialized_example,
                                                                                              is_pair)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            channel_out_uint8 = sess.run(node_channel_single_or_pair_uint8)
            coordinator.request_stop()
            coordinator.join(list_threads)
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/write_channel_single_or_pair/',
                                             tag_pair)
        
        # The directory containing the saved images is created
        # if it does not exist.
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        if is_pair:
            tls.save_image(os.path.join(path_to_directory_vis, 'channel_original.png'),
                           channel_out_uint8[:, :, 0])
            tls.save_image(os.path.join(path_to_directory_vis, 'channel_artifacts.png'),
                           channel_out_uint8[:, :, 1])
        else:
            tls.save_image(os.path.join(path_to_directory_vis, 'channel.png'),
                           numpy.squeeze(channel_out_uint8, axis=2))
    
    def test_write_context_portions_targets(self):
        """Tests the function `write_context_portions_targets` in the file "sets/writing.py".
        
        For i in {0, 1}, an image is saved at
        "sets/pseudo_visualization/write_context_portions_targets/width_target_64/image_i.png".
        The test is successful if, for i = 0, the
        image shows a dark gray target patch and its
        two bright gray context portions. For i = 1,
        the image shows a black target patch and its
        two black context portions.
        
        """
        width_target = 64
        path_to_tfrecord = 'sets/pseudo_data/write_context_portions_targets/data.tfrecord'
        
        portions_above_uint8 = 200*numpy.ones((2, width_target, 3*width_target, 1), dtype=numpy.uint8)
        portions_above_uint8[1, :, :, :] = 0
        portions_left_uint8 = 200*numpy.ones((2, 2*width_target, width_target, 1), dtype=numpy.uint8)
        portions_left_uint8[1, :, :, :] = 0
        targets_uint8 = 100*numpy.ones((2, width_target, width_target, 1), dtype=numpy.uint8)
        targets_uint8[1, :, :, :] = 0
        with tf.python_io.TFRecordWriter(path_to_tfrecord) as file:
            sets.writing.write_context_portions_targets(portions_above_uint8,
                                                        portions_left_uint8,
                                                        targets_uint8,
                                                        file)
        queue_tfrecord = tf.train.string_input_producer([path_to_tfrecord],
                                                        shuffle=False)
        
        # The 3rd argument of `sets.reading.read_queue_tfrecord` is a
        # tuple containing either one or two booleans.
        (node_portion_above_uint8, node_portion_left_uint8, node_target_uint8) = \
            sets.reading.read_queue_tfrecord(queue_tfrecord,
                                             width_target,
                                             (False,))
        path_to_directory_vis = os.path.join('sets/pseudo_visualization/write_context_portions_targets/',
                                             'width_target_{}'.format(width_target))
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            for i in range(2):
                [portion_above_uint8, portion_left_uint8, target_uint8] = sess.run(
                    [node_portion_above_uint8, node_portion_left_uint8, node_target_uint8]
                )
                image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                                      portion_left_uint8)
                image_uint8[width_target:2*width_target, width_target:2*width_target, :] = target_uint8
                tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                               numpy.squeeze(image_uint8, axis=2))
            coordinator.request_stop()
            coordinator.join(list_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests all the libraries in the directory "sets".')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterSets()
    getattr(tester, 'test_' + args.name)()


