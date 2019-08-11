"""A script to test all the libraries in the directory "ipfcns"."""

import argparse
import numpy
import os

import ipfcns.ipfcns
import tools.tools as tls

class TesterComparison(object):
    """Class for testing all the libraries in the directory "ipfcns"."""
    
    def test_arrange_pair_groups_lines(self):
        """Tests the function `arrange_pair_groups_lines`.
        
        An image is saved at
        "ipfcns/pseudo_visualization/arrange_pair_groups_lines.png".
        The test is successful if, in this image, there
        are 8 horizontal lines and 8 vertical lines forming
        a "L" flipped top to bottom.
        
        """
        width_target = 16
        
        group_lines_above_uint8 = numpy.zeros((8, 2*width_target + 8, 1),
                                              dtype=numpy.uint8)
        group_lines_left_uint8 = numpy.zeros((2*width_target, 8, 1),
                                             dtype=numpy.uint8)
        for i in range(8):
            group_lines_above_uint8[i, :, :] = i*10
            group_lines_left_uint8[:, i] = 100 + i*10
        image_uint8 = ipfcns.ipfcns.arrange_pair_groups_lines(group_lines_above_uint8,
                                                              group_lines_left_uint8)
        tls.save_image('ipfcns/pseudo_visualization/arrange_pair_groups_lines.png',
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_arrange_flattened_pair_groups_lines(self):
        """Tests the function `arrange_flattened_pair_groups_lines` in the file "ipfcns/ipfcns.py".
        
        An image is saved at
        "ipfcns/pseudo_visualization/arrange_flattened_pair_groups_lines.png".
        The test is successful if, in this image, the
        2nd row of the group of reference lines located
        above the target patch is in black and the 2nd last
        row of the group of reference lines located on the
        left side of the target patch is in dark gray.
        
        """
        width_target = 16
        
        flattened_pair_groups_lines_uint8 = 120*numpy.ones(32*width_target + 64,
                                                           dtype=numpy.uint8)
        flattened_pair_groups_lines_uint8[2*width_target + 8:4*width_target + 16] = 0
        flattened_pair_groups_lines_uint8[32*width_target + 48:32*width_target + 56] = 60
        image_uint8 = ipfcns.ipfcns.arrange_flattened_pair_groups_lines(flattened_pair_groups_lines_uint8,
                                                                        width_target)
        tls.save_image('ipfcns/pseudo_visualization/arrange_flattened_pair_groups_lines.png',
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_pair_groups_lines_from_channel(self):
        """Tests the function `extract_pair_groups_lines_from_channel` in the file "ipfcns/ipfcns.py".
        
        If `is_pair` is True, an image is saved at
        "ipfcns/pseudo_visualization/extract_pair_groups_lines_from_channel/width_target_64/pair/image.png".
        The test is successful if, in this image, the
        reference lines contain uniform noise.
        If `is_pair` is False, an image is saved at
        "ipfcns/pseudo_visualization/extract_pair_groups_lines_from_channel/width_target_64/single/image.png".
        The test is successful if both the reference
        lines have no uniform noise.
        
        """
        width_target = 64
        is_pair = False
        row_1st = 249
        col_1st = 349
        index_channel = 0
        
        rgb_uint8 = tls.read_image_mode('ipfcns/pseudo_data/rgb_cliff.jpg',
                                        'RGB')
        channel_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, index_channel:index_channel + 1]
        
        # If `is_pair` is True, an image channel and this image channel
        # with HEVC compression artifacts are inserted into
        # `ipfcns.ipfcns.extract_pair_groups_lines_from_channel`.
        # If `is_pair` is False, only this image channel is inserted
        # into `ipfcns.ipfcns.extract_pair_groups_lines_from_channel`.
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
        (group_lines_above_uint8, group_lines_left_uint8) = \
            ipfcns.ipfcns.extract_pair_groups_lines_from_channel(channel_single_or_pair_uint8,
                                                                 width_target,
                                                                 row_1st,
                                                                 col_1st)
        image_uint8 = ipfcns.ipfcns.arrange_pair_groups_lines(group_lines_above_uint8,
                                                              group_lines_left_uint8)
        path_to_directory_vis = os.path.join('ipfcns/pseudo_visualization/extract_pair_groups_lines_from_channel/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        tls.save_image(os.path.join(path_to_directory_vis, 'image.png'),
                       numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_pairs_groups_lines_from_channel(self):
        """Tests the function `extract_pairs_groups_lines_from_channel` in the file "ipfcns/ipfcns.py".
        
        If `is_pair` is True, for i = 0 ... 3, an image is saved at
        "ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channel/width_target_64/pair/image_i.png".
        The test is successful if, in each image, the reference
        lines contain uniform noise.
        If `is_pair` is False, for i = 0 ... 3, an image is saved at
        "ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channel/width_target_64/single/image_i.png".
        The test is successful if, in each image, the
        reference lines have no uniform noise.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([249, 249, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([349, 0, 349, 0], dtype=numpy.uint32)
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
        (groups_lines_above_uint8, groups_lines_left_uint8) = \
            ipfcns.ipfcns.extract_pairs_groups_lines_from_channel(channel_single_or_pair_uint8,
                                                                  width_target,
                                                                  row_1sts,
                                                                  col_1sts)
        path_to_directory_vis = os.path.join('ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channel/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # `groups_lines_above_uint8.shape[0]` and `groups_lines_left_uint8.shape[0]`
        # are equal to `row_1sts.size`.
        for i in range(groups_lines_above_uint8.shape[0]):
            image_uint8 = ipfcns.ipfcns.arrange_pair_groups_lines(groups_lines_above_uint8[i, :, :, :],
                                                                  groups_lines_left_uint8[i, :, :, :])
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_pairs_groups_lines_from_channels(self):
        """Tests the function `extract_pairs_groups_lines_from_channels` in the file "ipfcns/ipfcns.py".
        
        If `is_pair` is True, for i = 0 ... 7, an image is saved at
        "ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channels/width_target_64/pair/image_i.png".
        The test is successful if, in each image, the reference
        lines contain uniform noise.
        If `is_pair` is False, for i = 0 ... 7, an image is saved at
        "ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channels/width_target_64/single/image_i.png".
        The test is successful if, in each image, the reference
        lines have no uniform noise. Besides, whatever the value
        of `is_pair`, for i = 0 ... 3, the reference lines come
        from an image channel. For i = 4 ... 7, the reference
        lines come from the same channel of another image.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([249, 249, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([349, 0, 349, 0], dtype=numpy.uint32)
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
        (groups_lines_above_uint8, groups_lines_left_uint8) = \
            ipfcns.ipfcns.extract_pairs_groups_lines_from_channels(channels_single_or_pair_uint8,
                                                                   width_target,
                                                                   row_1sts,
                                                                   col_1sts)
        path_to_directory_vis = os.path.join('ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channels/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # `groups_lines_above_uint8.shape[0]` and `groups_lines_left_uint8.shape[0]`
        # are equal to `2*row_1sts.size`.
        for i in range(groups_lines_above_uint8.shape[0]):
            image_uint8 = ipfcns.ipfcns.arrange_pair_groups_lines(groups_lines_above_uint8[i, :, :, :],
                                                                  groups_lines_left_uint8[i, :, :, :])
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_extract_pairs_groups_lines_from_channels_plus_preprocessing(self):
        """Tests the function `extract_pairs_groups_lines_from_channels_plus_preprocessing` in the file "ipfcns/ipfcns.py".
        
        If `is_pair` is True, for i = 0 ... 7, an image is saved at
        "ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channels_plus_preprocessing/width_target_64/pair/image_i.png".
        If `is_pair` is False, for i = 0 ... 7, an image is saved at
        "ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channels_plus_preprocessing/width_target_64/single/image_i.png".
        The test is successful if, in each image, there is
        no degradation due to the fact that a forward preprocessing
        and an inverse preprocessing were carried out.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([249, 249, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([349, 0, 349, 0], dtype=numpy.uint32)
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
        (flattened_pairs_groups_lines_float32, means_float32) = \
            ipfcns.ipfcns.extract_pairs_groups_lines_from_channels_plus_preprocessing(channels_single_or_pair_uint8,
                                                                                      width_target,
                                                                                      row_1sts,
                                                                                      col_1sts)
        path_to_directory_vis = os.path.join('ipfcns/pseudo_visualization/extract_pairs_groups_lines_from_channels_plus_preprocessing/',
                                             'width_target_{}'.format(width_target),
                                             tag_pair)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # `flattened_pairs_groups_lines_float32.shape[0]` is equal to `means_float32.size`.
        for i in range(flattened_pairs_groups_lines_float32.shape[0]):
            flattened_pair_groups_lines_uint8 = tls.cast_float_to_uint8(flattened_pairs_groups_lines_float32[i, :] + means_float32[i])
            image_uint8 = ipfcns.ipfcns.arrange_flattened_pair_groups_lines(flattened_pair_groups_lines_uint8,
                                                                            width_target)
            tls.save_image(os.path.join(path_to_directory_vis, 'image_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
    
    def test_predict_by_batch_via_ipfcns(self):
        """Tests the function `predict_by_batch_via_ipfcns` in the file "ipfcns/ipfcns.py".
        
        For i = 0 ... 7, two images are saved at respectively
        "ipfcns/pseudo_visualization/predict_by_batch_via_ipfcns/width_target_16/reference_lines_i.png"
        and "ipfcns/pseudo_visualization/predict_by_batch_via_ipfcns/width_target_16/prediction_i.png".
        The test is successful if, for each i, the reference lines
        in the 1st image are consistent with the prediction in the
        2nd image.
        
        """
        # Caffe is imported for the current test only.
        # This way, the other tests can be run without
        # importing Caffe.
        caffe = __import__('caffe')
        
        # `batch_size` is equal to 4 because the 1st input
        # dimension written in the file "ipfcns/pseudo_data/predict_by_batch_via_ipfcns/width_target_16/IntraFCN205_deploy_Size16.prototxt"
        # is equal to 4.
        batch_size = 4
        width_target = 16
        row_1sts = numpy.array([249, 249, 0, 0],
                               dtype=numpy.int16)
        col_1sts = numpy.array([349, 0, 349, 0],
                               dtype=numpy.uint32)
        index_channel = 0
        
        path_to_directory_load = 'ipfcns/pseudo_data/predict_by_batch_via_ipfcns/width_target_{}/'.format(width_target)
        net_ipfcns = caffe.Net(os.path.join(path_to_directory_load, 'IntraFCN205_deploy_Size{}.prototxt'.format(width_target)),
                               os.path.join(path_to_directory_load, 'IntraFCN205_Size{}_iter_1638700.caffemodel'.format(width_target)),
                              caffe.TEST)
        
        rgb_0_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        channel_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[:, :, index_channel:index_channel + 1]
        rgb_1_uint8 = tls.read_image_mode('sets/pseudo_data/rgb_library.jpg',
                                          'RGB')
        channel_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[:, :, index_channel:index_channel + 1]
        channels_single_or_pair_uint8 = numpy.stack((channel_0_uint8, channel_1_uint8),
                                                    axis=0)
        (flattened_pairs_groups_lines_float32, means_float32) = \
            ipfcns.ipfcns.extract_pairs_groups_lines_from_channels_plus_preprocessing(channels_single_or_pair_uint8,
                                                                                      width_target,
                                                                                      row_1sts,
                                                                                      col_1sts)
        predictions_float32 = ipfcns.ipfcns.predict_by_batch_via_ipfcns(flattened_pairs_groups_lines_float32,
                                                                        net_ipfcns,
                                                                        width_target,
                                                                        batch_size)
        path_to_directory_vis = os.path.join('ipfcns/pseudo_visualization/predict_by_batch_via_ipfcns/',
                                             'width_target_{}'.format(width_target))
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        for i in range(predictions_float32.shape[0]):
            flattened_pair_groups_lines_uint8 = tls.cast_float_to_uint8(flattened_pairs_groups_lines_float32[i, :] + means_float32[i])
            image_uint8 = ipfcns.ipfcns.arrange_flattened_pair_groups_lines(flattened_pair_groups_lines_uint8,
                                                                            width_target)
            tls.save_image(os.path.join(path_to_directory_vis, 'reference_lines_{}.png'.format(i)),
                           numpy.squeeze(image_uint8, axis=2))
            prediction_uint8 = tls.cast_float_to_uint8(predictions_float32[i, :, :, :] + means_float32[i])
            tls.save_image(os.path.join(path_to_directory_vis, 'prediction_{}.png'.format(i)),
                           numpy.squeeze(prediction_uint8, axis=2))
    
    def test_preprocess_pairs_groups_lines(self):
        """Tests the function `preprocess_pairs_groups_lines` in the file "ipfcns/ipfcns.py".
        
        The test is successful if the mean of each row of
        the flattened reference lines array is close to 0.0.
        
        """
        width_target = 64
        is_pair = False
        row_1sts = numpy.array([249, 249, 0, 0], dtype=numpy.int16)
        col_1sts = numpy.array([349, 0, 349, 0], dtype=numpy.uint32)
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
        (groups_lines_above_uint8, groups_lines_left_uint8) = \
            ipfcns.ipfcns.extract_pairs_groups_lines_from_channels(channels_single_or_pair_uint8,
                                                                   width_target,
                                                                   row_1sts,
                                                                   col_1sts)
        flattened_pairs_groups_lines_float32 = ipfcns.ipfcns.preprocess_pairs_groups_lines(groups_lines_above_uint8,
                                                                                           groups_lines_left_uint8)[0]
        print('Mean of each row of the flattened reference lines array:')
        print(numpy.mean(flattened_pairs_groups_lines_float32, axis=1))
    
    def test_visualize_flattened_pair_groups_lines(self):
        """Tests the function `visualize_flattened_pair_groups_lines` in the file "ipfcns/ipfcns.py".
        
        An image is saved at
        "ipfcns/pseudo_visualization/visualize_flattened_pair_groups_lines.png".
        The test is successful if this image
        shows a gray context whose 2nd row
        is black. The background is white.
        
        """
        width_target = 8
        mean_pair_groups_lines = 110.2
        
        flattened_pair_groups_lines_float32 = numpy.zeros(64 + 32*width_target, dtype=numpy.float32)
        flattened_pair_groups_lines_float32[8 + 2*width_target:16 + 4*width_target] = -mean_pair_groups_lines
        ipfcns.ipfcns.visualize_flattened_pair_groups_lines(flattened_pair_groups_lines_float32,
                                                            width_target,
                                                            mean_pair_groups_lines,
                                                            'ipfcns/pseudo_visualization/visualize_flattened_pair_groups_lines.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests all the libraries in the directory "ipfcns".')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterComparison()
    getattr(tester, 'test_' + args.name)()


