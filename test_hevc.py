"""A script to test all the libraries in the directory "hevc"."""

import argparse
import filecmp
import numpy
import os

import hevc.constants
import hevc.intraprediction.intraprediction
import hevc.performance
import hevc.running
import hevc.stats
import hevc.unifiedloading
import tools.tools as tls


class TesterHEVC(object):
    """Class for testing all the libraries in the directory "hevc"."""
    
    def test_collect_computation_time(self):
        """Tests the function `collect_computation_time` in the file "hevc/performance.py".
        
        The test is successful if the collected computation
        time is equal to the computation time in the file at
        "hevc/pseudo_data/collect_computation_time/pseudo_log.txt".
        
        """
        path_to_log_encoder_or_decoder = 'hevc/pseudo_data/collect_computation_time/pseudo_log.txt'
        
        computation_time = hevc.performance.collect_computation_time(path_to_log_encoder_or_decoder)
        print('Collected computation time: {} seconds.'.format(computation_time))
    
    def test_collect_nb_bits_psnrs_from_log_encoder(self):
        """Tests the function `collect_nb_bits_psnrs_from_log_encoder` in the file "hevc/performance.py".
        
        The test is successful if the collected number of bits
        and the collected PSNRs correspond to the number of bits
        and the PSNRs in the file at "hevc/pseudo_data/collect_nb_bits_psnrs_from_log_encoder/pseudo_log.txt".
        
        """
        path_to_log_encoder = 'hevc/pseudo_data/collect_nb_bits_psnrs_from_log_encoder/pseudo_log.txt'
        
        (nb_bits, psnrs) = hevc.performance.collect_nb_bits_psnrs_from_log_encoder(path_to_log_encoder)
        print('Collected number of bits: {}'.format(nb_bits))
        print('Collected PSNRs:')
        print(psnrs)
    
    def test_compute_rate_psnr(self):
        """Tests the function `compute_rate_psnr` in the file "hevc/performance.py".
        
        The test is successful if the rate of the encoding and decoding
        of the image via HEVC computed by `compute_rate_psnr` is equal to
        the expected rate. Besides, the PSNR of the encoding and decoding
        of the image via HEVC computed by `compute_rate_psnr` must be equal
        to the expected PSNR.
        
        """
        nb_channels = 1
        path_to_directory_temp = 'hevc/pseudo_data/temp/compute_rate_psnr/'
        path_to_before_encoding_hevc = os.path.join(path_to_directory_temp,
                                                    'image_before_encoding_hevc.yuv')
        path_to_after_encoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_encoding_hevc.yuv')
        path_to_after_decoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_decoding_hevc.yuv')
        if nb_channels == 1:
            path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
        elif nb_channels == 3:
            path_to_cfg = 'hevc/configuration/intra_main.cfg'
        else:
            raise ValueError('`nb_channels` is equal to neither 1 or 3.')
        path_to_bitstream = os.path.join(path_to_directory_temp,
                                         'bitstream.bin')
        path_to_log_encoder = os.path.join(path_to_directory_temp,
                                           'log_encoder.txt')
        path_to_log_decoder = os.path.join(path_to_directory_temp,
                                           'log_decoder.txt')
        qp = 42
        
        rgb_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                        'RGB')
        
        # The first two dimensions of `image_before_encoding_hevc_uint8` are
        # divisible by 8.
        image_before_encoding_hevc_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[0:384, 0:384, 0:nb_channels]
        (_, rate, psnrs, computation_times) = \
            hevc.performance.compute_rate_psnr(image_before_encoding_hevc_uint8,
                                               path_to_before_encoding_hevc,
                                               path_to_after_encoding_hevc,
                                               path_to_after_decoding_hevc,
                                               path_to_cfg,
                                               path_to_bitstream,
                                               hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                               hevc.constants.PATH_TO_EXE_DECODER_REGULAR,
                                               path_to_log_encoder,
                                               path_to_log_decoder,
                                               qp,
                                               None)
        print('Rate of the encoding and decoding of the image via HEVC computed by `compute_rate_psnr`: {} bbp.'.format(rate))
        print('PSNRs of the encoding and decoding of the image via HEVC computed by `compute_rate_psnr`: {} dB.'.format(psnrs))
        if nb_channels == 1:
            print('Expected rate: 0.146647135417 bbp.')
            print('Expected PSNRs: [28.2142 0. 0.] dB.')
        print('Computation time of the encoding: {} seconds.'.format(computation_times[0]))
        print('Computation time of the decoding: {} seconds.'.format(computation_times[1]))
    
    def test_compute_rates_psnrs(self):
        """Tests the function `compute_rates_psnrs` in the file "hevc/performance.py".
        
        The test is successful if, when the quantization parameter
        is equal to 12, the rate is relatively large and the PSNRs too.
        When the quantization parameter is equal to 42, the rate is
        relatively small and the PSNRs too.
        
        """
        nb_channels = 3
        path_to_directory_temp = 'hevc/pseudo_data/temp/compute_rates_psnrs/'
        path_to_before_encoding_hevc = os.path.join(path_to_directory_temp,
                                                    'image_before_encoding_hevc.yuv')
        path_to_after_encoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_encoding_hevc.yuv')
        path_to_after_decoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_decoding_hevc.yuv')
        if nb_channels == 1:
            path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
            tag_400_420 = '400'
        elif nb_channels == 3:
            path_to_cfg = 'hevc/configuration/intra_main.cfg'
            tag_400_420 = '420'
        else:
            raise ValueError('`nb_channels` is equal to neither 1 or 3.')
        path_to_bitstream = os.path.join(path_to_directory_temp,
                                         'bitstream.bin')
        path_to_log_encoder = os.path.join(path_to_directory_temp,
                                           'log_encoder.txt')
        path_to_log_decoder = os.path.join(path_to_directory_temp,
                                           'log_decoder.txt')
        qps_int = numpy.array([12, 42],
                              dtype=numpy.int32)
        
        # `paths_to_saved_rotated_images_crops` contains the paths
        # to the saved visualizations for all the reconstructed images
        # after the decoding via HEVC for all quantization parameters.
        paths_to_saved_rotated_images_crops = []
        for qp in qps_int:
            paths_temp = []
            path_to_directory_qp = os.path.join('hevc/pseudo_visualization/compute_rates_psnrs/',
                                                tag_400_420,
                                                'qp_{}'.format(qp))
            if not os.path.isdir(path_to_directory_qp):
                os.makedirs(path_to_directory_qp)
            for tag_image in ('library', 'cliff'):
                paths_temp.append(
                    [
                        [os.path.join(path_to_directory_qp, '{0}_reconstruction_{1}.png'.format(tag_image, i)) for i in range(nb_channels)],
                        [os.path.join(path_to_directory_qp, '{0}_crop0_{1}.png'.format(tag_image, i)) for i in range(nb_channels)],
                        [os.path.join(path_to_directory_qp, '{0}_crop1_{1}.png'.format(tag_image, i)) for i in range(nb_channels)]
                    ]
                )
            
            # A list of lists of lists of paths is appended
            # to `paths_to_saved_rotated_images_crops`. 
            paths_to_saved_rotated_images_crops.append(paths_temp)
        
        # If `dict_visualization` is not None,
        # `hevc.performance.compute_rates_psnrs`
        # saves visualizations.
        dict_visualization = {
            'rows_columns_top_left': numpy.array([[0, 300], [200, 0]], dtype=numpy.uint16),
            'width_crop': 80,
            'list_indices_rotation': [],
            'paths_to_saved_rotated_images_crops': paths_to_saved_rotated_images_crops
        }
        
        rgb_0_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                          'RGB')
        rgb_1_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        image_before_encoding_hevc_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[0:512, 0:512, 0:nb_channels]
        image_before_encoding_hevc_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[0:512, 0:512, 0:nb_channels]
        images_before_encoding_hevc_uint8 = numpy.stack((image_before_encoding_hevc_0_uint8, image_before_encoding_hevc_1_uint8),
                                                        axis=0)
        (rates, psnrs, computation_times) = \
            hevc.performance.compute_rates_psnrs(images_before_encoding_hevc_uint8,
                                                 path_to_before_encoding_hevc,
                                                 path_to_after_encoding_hevc,
                                                 path_to_after_decoding_hevc,
                                                 path_to_cfg,
                                                 path_to_bitstream,
                                                 hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                                 hevc.constants.PATH_TO_EXE_DECODER_REGULAR,
                                                 path_to_log_encoder,
                                                 path_to_log_decoder,
                                                 qps_int,
                                                 None,
                                                 dict_visualization)
        for i in range(qps_int.size):
            print('Quantization parameter: {}.'.format(qps_int[i]))
            for j in range(images_before_encoding_hevc_uint8.shape[0]):
                print('Image of index {}.'.format(j))
                print('Rate: {} bbp.'.format(rates[i, j]))
                print('PSNRs: {} dB.'.format(psnrs[i, j, :]))
                print('Computation time of the encoding: {} seconds.'.format(computation_times[i, j, 0]))
                print('Computation time of the decoding: {} seconds.'.format(computation_times[i, j, 1]))
    
    def test_convert_accumulations_into_ratios(self):
        """Tests the function `convert_accumulations_into_ratios` in the file "hevc/stats.py".
        
        The test is successful if, for each kind of statistics,
        the ratios computed by the function are similar to the
        expected ratios.
        
        """
        accumulations_0_int64 = numpy.array([[3, 3, 6], [7, 3, 0]],
                                            dtype=numpy.int64)
        accumulations_1_int64 = numpy.array([[1, 6, 3], [9, 0, 3]],
                                            dtype=numpy.int64)
        accumulations_with_index_mode_int64 = numpy.stack((accumulations_0_int64, accumulations_1_int64),
                                                          axis=0)
        counts_int16 = numpy.array([10, 6, 6],
                                   dtype=numpy.int16)
        ratios_float64 = hevc.stats.convert_accumulations_into_ratios(accumulations_with_index_mode_int64,
                                                                      counts_int16)
        print('Ratios for the 1st kind of statistics computed by `convert_accumulations_into_ratios`:')
        print(ratios_float64[0, :, :])
        print('Expected ratios for the 1st kind of statistics:')
        print(numpy.array([[0.3, 0.5, 1.], [0.7, 0.5, 0.]]))
        print('Ratios for the 2nd kind of statistics computed by the function:')
        print(ratios_float64[1, :, :])
        print('Expected ratios for the 2nd kind of statistics:')
        print(numpy.array([[0.1, 1., 0.5], [0.9, 0., 0.5]]))
    
    def test_convert_nb_bits_into_rate(self):
        """Tests the function `convert_nb_bits_into_rate` in the file "hevc/performance.py".
        
        The test is successful if the rate is equal to 0.001 bbp.
        
        """
        nb_bits = 1000
        shape_image = (500, 2000, 1)
        
        rate = hevc.performance.convert_nb_bits_into_rate(nb_bits,
                                                          shape_image)
        print('Shape of the image: {}'.format(shape_image))
        print('Number of bits in the bitstream: {}'.format(nb_bits))
        print('Rate: {} bbp.'.format(rate))
    
    def test_encode_decode_image(self):
        """Tests the function `encode_decode_image` in the file "hevc/running.py".
        
        A 1st image is saved at
        "hevc/pseudo_visualization/encode_decode_image/400/image_before_encoding_hevc_0.png".
        A 2nd image is saved at
        "hevc/pseudo_visualization/encode_decode_image/400/reconstructed_image_after_decoding_hevc_0.png".
        The test is successful if the 2nd
        image corresponds to the 1st image
        with compression artefacts.
        
        """
        nb_channels = 3
        path_to_directory_temp = 'hevc/pseudo_data/temp/encode_decode_image/'
        path_to_before_encoding_hevc = os.path.join(path_to_directory_temp,
                                                    'image_before_encoding_hevc.yuv')
        path_to_after_encoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_encoding_hevc.yuv')
        path_to_after_decoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_decoding_hevc.yuv')
        if nb_channels == 1:
            path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
            tag_400_420 = '400'
        elif nb_channels == 3:
            path_to_cfg = 'hevc/configuration/intra_main.cfg'
            tag_400_420 = '420'
        else:
            raise ValueError('`nb_channels` does not belong to {1, 3}.')
        path_to_bitstream = os.path.join(path_to_directory_temp,
                                         'bitstream.bin')
        path_to_log_encoder = os.path.join(path_to_directory_temp,
                                           'log_encoder.txt')
        path_to_log_decoder = os.path.join(path_to_directory_temp,
                                           'log_decoder.txt')
        qp = 42
        path_to_directory_vis = os.path.join('hevc/pseudo_visualization/encode_decode_image/',
                                             tag_400_420)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        rgb_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                        'RGB')
        
        # HEVC only encodes images whose height and width are divisible
        # by the minimum CY size, i.e. 8 pixels.
        height_divisible_by_8 = 8*(rgb_uint8.shape[0]//8)
        width_divisible_by_8 = 8*(rgb_uint8.shape[1]//8)
        image_before_encoding_hevc_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[0:height_divisible_by_8, 0:width_divisible_by_8, 0:nb_channels]
        
        # The last argument of `encode_decode_image` is
        # None as no prediction neural network is used
        # for the current test.
        reconstructed_image_after_decoding_hevc_uint8 = hevc.running.encode_decode_image(image_before_encoding_hevc_uint8,
                                                                                         path_to_before_encoding_hevc,
                                                                                         path_to_after_encoding_hevc,
                                                                                         path_to_after_decoding_hevc,
                                                                                         path_to_cfg,
                                                                                         path_to_bitstream,
                                                                                         hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                                                                         hevc.constants.PATH_TO_EXE_DECODER_REGULAR,
                                                                                         path_to_log_encoder,
                                                                                         path_to_log_decoder,
                                                                                         qp,
                                                                                         None)
        
        # In the current test, the HEVC encoder log file and
        # the HEVC decoder log file are not useful.
        os.remove(path_to_log_encoder)
        os.remove(path_to_log_decoder)
        for i in range(nb_channels):
            tls.save_image(os.path.join(path_to_directory_vis, 'channel_before_encoding_hevc_{}.png'.format(i)),
                           image_before_encoding_hevc_uint8[:, :, i])
            tls.save_image(os.path.join(path_to_directory_vis, 'reconstructed_channel_after_decoding_hevc_{}.png'.format(i)),
                           reconstructed_image_after_decoding_hevc_uint8[:, :, i])
    
    def test_encode_image(self):
        """Tests the function `encode_image` in the file "hevc/running.py".
        
        A 1st image is saved at
        "hevc/pseudo_visualization/encode_image/400/image_before_encoding_hevc_0.png".
        A 2nd image is saved at
        "hevc/pseudo_visualization/encode_image/400/reconstructed_image_after_encoding_hevc_0.png".
        The test is successful if the 2nd image
        corresponds to the 1st image with HEVC
        compression artifacts.
        
        """
        # Thanks to `nb_channels`, `encode_image` can be tested
        # using either a luminance image or a YCbCr image.
        nb_channels = 3
        path_to_directory_temp = 'hevc/pseudo_data/temp/encode_image/'
        path_to_before_encoding_hevc = os.path.join(path_to_directory_temp,
                                                    'image_before_encoding_hevc.yuv')
        path_to_after_encoding_hevc = os.path.join(path_to_directory_temp,
                                                   'image_after_encoding_hevc.yuv')
        if nb_channels == 1:
            path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
            tag_400_420 = '400'
        elif nb_channels == 3:
            path_to_cfg = 'hevc/configuration/intra_main.cfg'
            tag_400_420 = '420'
        else:
            raise ValueError('`nb_channels` does not belong to {1, 3}.')
        path_to_bitstream = os.path.join(path_to_directory_temp,
                                         'bitstream.bin')
        qp = 42
        path_to_directory_vis = os.path.join('hevc/pseudo_visualization/encode_image/',
                                             tag_400_420)
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        
        # `rgb_uint8.shape[0]` is not divisible by 2. That is
        # why `rgb_uint8` is cropped.
        rgb_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                        'RGB')
        height_divisible_by_8 = 8*(rgb_uint8.shape[0]//8)
        width_divisible_by_8 = 8*(rgb_uint8.shape[1]//8)
        image_before_encoding_hevc_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[0:height_divisible_by_8, 0:width_divisible_by_8, 0:nb_channels]
        reconstructed_image_after_encoding_hevc_uint8 = hevc.running.encode_image(image_before_encoding_hevc_uint8,
                                                                                  path_to_before_encoding_hevc,
                                                                                  path_to_after_encoding_hevc,
                                                                                  path_to_cfg,
                                                                                  path_to_bitstream,
                                                                                  hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                                                                  qp,
                                                                                  None)
        for i in range(nb_channels):
            tls.save_image(os.path.join(path_to_directory_vis, 'channel_before_encoding_hevc_{}.png'.format(i)),
                           image_before_encoding_hevc_uint8[:, :, i])
            tls.save_image(os.path.join(path_to_directory_vis, 'reconstructed_channel_after_encoding_hevc_{}.png'.format(i)),
                           reconstructed_image_after_encoding_hevc_uint8[:, :, i])
    
    def test_encode_luminances_extract_statistics_from_files(self):
        """Tests the function `encode_luminances_extract_statistics_from_files` in the file "hevc/stats.py".
        
        The test is successful if, for each width of target patch,
        the frequency of winning the fast selection sums to 1.0 over
        all modes. Besides, for each width of target patch, the frequency
        of being found in the fast list does not sum to 1.0 over all
        modes.
        
        """
        path_to_directory_temp = 'hevc/pseudo_data/temp/encode_luminances_extract_statistics_from_files/'
        path_to_before_encoding_hevc = os.path.join(path_to_directory_temp,
                                                    'luminance_before_encoding_hevc.yuv')
        path_to_after_encoding_hevc = os.path.join(path_to_directory_temp,
                                                   'luminance_after_encoding_hevc.yuv')
        path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
        path_to_bitstream = os.path.join(path_to_directory_temp,
                                         'bitstream.bin')
        qp = 32
        paths_to_stats = [
            'hevc/pseudo_data/encode_luminances_extract_statistics_from_files/stats_library.txt',
            'hevc/pseudo_data/encode_luminances_extract_statistics_from_files/stats_cliff.txt'
        ]
        pairs_beacons = (
            ('index', 'wins the fast selection:'),
            ('index', 'is found in the fast list:')
        )
        beacon_run = '{fast selection, rate-distortion selection} is run:'
        
        # If `nb_modes` is strictly smaller than the true number
        # of intra prediction modes in HEVC, an exception will be
        # raised during the extraction of statistics.
        nb_modes = 35
        
        # If the quatree partitioning in HEVC is allowed to go
        # down to 4x4 Prediction Blocks (PBs), `nb_widths_target`
        # has to be equal to 5. Otherwise, an exception will be
        # raised during the extraction of statistics.
        nb_widths_target = 5
        
        rgb_0_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                          'RGB')
        luminance_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[0:512, 0:512, 0:1]
        rgb_1_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        luminance_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[0:512, 0:512, 0:1]
        luminances_uint8 = numpy.stack((luminance_0_uint8, luminance_1_uint8),
                                       axis=0)
        ratios_float64 = hevc.stats.encode_luminances_extract_statistics_from_files(luminances_uint8,
                                                                                    path_to_before_encoding_hevc,
                                                                                    path_to_after_encoding_hevc,
                                                                                    path_to_cfg,
                                                                                    path_to_bitstream,
                                                                                    hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                                                                    qp,
                                                                                    None,
                                                                                    paths_to_stats,
                                                                                    pairs_beacons,
                                                                                    beacon_run,
                                                                                    nb_modes,
                                                                                    nb_widths_target)
        sums_wins_fast_selection_float64 = numpy.sum(ratios_float64[0, :, :],
                                                     axis=0)
        sums_found_in_fast_list_float64 = numpy.sum(ratios_float64[1, :, :],
                                                    axis=0)
        print('Frequency of winning the fast selection summed over all modes:')
        print(sums_wins_fast_selection_float64)
        print('Frequency of being found in the fast list summed over all modes:')
        print(sums_found_in_fast_list_float64)
    
    def test_extract_intra_pattern(self):
        """Tests the function `extract_intra_pattern` in the file "hevc/intraprediction/intraprediction.py".
        
        A 1st image is saved at
        "hevc/pseudo_visualization/extract_intra_pattern/channel.png".
        A 2nd image is saved at
        "hevc/pseudo_visualization/extract_intra_pattern/intra_pattern.png".
        The test is successful if the top part and the
        left part of the bright pattern in the 1st image
        is placed at respectively the top and the left of
        the 2nd image.
        
        """
        width_target = 32
        row_ref = 5
        col_ref = 10
        tuple_width_height_masks = (4, 32)
        
        channel_uint8 = 50*numpy.ones((200, 200, 1),
                                      dtype=numpy.uint8)
        channel_uint8[row_ref:, col_ref] = 250
        channel_uint8[row_ref, col_ref:] = 150
        intra_pattern_uint8 = hevc.intraprediction.intraprediction.extract_intra_pattern(channel_uint8,
                                                                                         width_target,
                                                                                         row_ref,
                                                                                         col_ref,
                                                                                         tuple_width_height_masks)
        tls.save_image('hevc/pseudo_visualization/extract_intra_pattern/channel.png',
                       numpy.squeeze(channel_uint8, axis=2))
        tls.save_image('hevc/pseudo_visualization/extract_intra_pattern/intra_pattern.png',
                       numpy.squeeze(intra_pattern_uint8, axis=2))
    
    def test_extract_intra_patterns(self):
        """Tests the function `extract_intra_patterns` in the file "hevc/intraprediction/intraprediction.py".
        
        For i = 0 ... 3, an image is saved at
        "hevc/pseudo_visualization/extract_intra_patterns/intra_pattern_i.png".
        The test is successful if, in each image, all
        the pixels, excluding those in the 1st row and
        those in the 1st column, are in white.
        
        """
        width_target = 32
        row_refs = numpy.array([0, 580],
                               dtype=numpy.int16)
        col_refs = numpy.array([0, 911],
                               dtype=numpy.uint32)
        tuple_width_height_masks = (16, 4)
        index_channel = 0
        
        rgb_0_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                          'RGB')
        channel_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[:, :, index_channel:index_channel + 1]
        rgb_1_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        channel_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[:, :, index_channel:index_channel + 1]
        channels_uint8 = numpy.stack((channel_0_uint8, channel_1_uint8),
                                     axis=0)
        intra_patterns_uint8 = hevc.intraprediction.intraprediction.extract_intra_patterns(channels_uint8,
                                                                                           width_target,
                                                                                           row_refs,
                                                                                           col_refs,
                                                                                           tuple_width_height_masks)
        for i in range(intra_patterns_uint8.shape[0]):
            tls.save_image('hevc/pseudo_visualization/extract_intra_patterns/intra_pattern_{}.png'.format(i),
                           numpy.squeeze(intra_patterns_uint8[i, :, :, :], axis=2))
    
    def test_extract_statistics_from_files(self):
        """Tests the function `extract_statistics_from_files` in the file "hevc/stats.py".
        
        The test is successful if the statistics computed
        by the function are similar to the expected statistics.
        
        """
        paths_to_stats = [
            'hevc/pseudo_data/pseudo_text_0.txt',
            'hevc/pseudo_data/pseudo_text_1.txt'
        ]
        pairs_beacons = (
            ('beacon_0', 'beacon_1'),
            ('bli_0', 'bli_1')
        )
        beacons_before_integers = ('bloc_0', 'bea_0')
        nb_modes = 4
        nb_widths_target = 4
        
        # In the text files, the statistics are not sorted
        # according to the mode index.
        (accumulations_with_index_mode_int64, accumulations_without_index_mode_int64) = \
            hevc.stats.extract_statistics_from_files(paths_to_stats,
                                                     pairs_beacons,
                                                     beacons_before_integers,
                                                     nb_modes,
                                                     nb_widths_target)
        print('1st series of statistics depending on the HEVC intra prediction mode computed by `extract_statistics_from_files`:')
        print(accumulations_with_index_mode_int64[0, :, :])
        print('Expected 1st series of statistics depending on the HEVC intra prediction mode:')
        print(numpy.array([[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22], [0, -2, -4, -6]], dtype=numpy.int64))
        print('2nd series of statistics depending on the HEVC intra prediction mode computed by `extract_statistics_from_files`:')
        print(accumulations_with_index_mode_int64[1, :, :])
        print('Expected 2nd series of statistics depending on the HEVC intra prediction mode:')
        print(numpy.zeros((nb_modes, nb_widths_target), dtype=numpy.int64))
        print('1st series of statistics that do not depend on the HEVC intra prediction mode computed by `extract_statistics_from_files`:')
        print(accumulations_without_index_mode_int64[0, :])
        print('Expected 1st series of statistics that do not depend on the HEVC intra prediction mode:')
        print(numpy.zeros(nb_widths_target, dtype=numpy.int64))
        
        # In the text file "hevc/pseudo_data/pseudo_text_1.txt", there
        # are three beacons "bea_0". The indicators associated to the
        # the 3rd beacon "bea_0" overwrite the indicators associated to
        # the first two beacons "bea_0".
        print('2nd series of statistics that do not depend on the HEVC intra prediction mode computed by `extract_statistics_from_files`:')
        print(accumulations_without_index_mode_int64[1, :])
        print('Expected 2nd series of statistics that do not depend on the HEVC intra prediction mode:')
        print(numpy.array([-1, -10, -10, -1], dtype=numpy.int64))
    
    def test_fill_if_beacon_found(self):
        """Tests the function `fill_if_beacon_found` in the file "hevc/stats.py".
        
        The test is successful if the container after it is filled
        by the function is similar to the expected container.
        
        """
        beacon_before_integers = 'beacon_1 0'
        
        container_1d_int64 = numpy.zeros(3, dtype=numpy.int64)
        print('Container before it is filled by `fill_if_beacon_found`:')
        print(container_1d_int64)
        with open('hevc/pseudo_data/pseudo_text_0.txt', 'r') as file:
            for line_text in file:
                hevc.stats.fill_if_beacon_found(line_text,
                                                beacon_before_integers,
                                                container_1d_int64)
        print('Container after it is filled by `fill_if_beacon_found`:')
        print(container_1d_int64)
        print('Expected container:')
        print(numpy.array([-1, -2, -3], dtype=numpy.int64))
    
    def test_fill_if_beacons_found(self):
        """Tests the function `fill_if_beacons_found` in the file "hevc/stats.py".
        
        The test is successful if the container after it is filled
        by the function is similar to the expected container.
        
        """
        beacon_before_index_row = 'beacon_0'
        beacon_after_index_row = 'beacon_1'
        
        container_2d_int64 = numpy.zeros((4, 4), dtype=numpy.int64)
        print('Container before it is filled by `fill_if_beacons_found`:')
        print(container_2d_int64)
        with open('hevc/pseudo_data/pseudo_text_0.txt', 'r') as file:
            for line_text in file:
                hevc.stats.fill_if_beacons_found(line_text,
                                                 beacon_before_index_row,
                                                 beacon_after_index_row,
                                                 container_2d_int64)
        print('Container after it is filled by `fill_if_beacons_found`:')
        print(container_2d_int64)
        print('Expected container:')
        print(numpy.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, -1, -2, -3]], dtype=numpy.int64))
    
    def test_find_load_grayscale(self):
        """Tests the function `find_load_grayscale` in the file "hevc/unifiedloading.py".
        
        The test is successful if the saved image "hevc/pseudo_visualization/find_load_grayscale.png"
        is the image at "hevc/pseudo_data/new_york.jpg".
        
        """
        path_to_directory_file = 'hevc/pseudo_data/'
        prefix_filename = 'new_york'
        
        grayscale_uint8 = hevc.unifiedloading.find_load_grayscale(path_to_directory_file,
                                                                  prefix_filename)
        tls.save_image('hevc/pseudo_visualization/find_load_grayscale.png',
                       numpy.squeeze(grayscale_uint8, axis=2))
    
    def test_find_rgb_load_luminance(self):
        """Tests the function `find_rgb_load_luminance` in the file "hevc/unifiedloading.py".
        
        The test is successful if the saved image "hevc/pseudo_visualization/find_rgb_load_luminance.png"
        is the luminance version of the loaded RGB image.
        
        """
        path_to_directory_file = 'hevc/pseudo_data/'
        prefix_filename = 'rgb_library'
        
        luminance_uint8 = hevc.unifiedloading.find_rgb_load_luminance(path_to_directory_file,
                                                                      prefix_filename)
        tls.save_image('hevc/pseudo_visualization/find_rgb_load_luminance.png',
                       numpy.squeeze(luminance_uint8, axis=2))
    
    def test_find_video_load_luminance(self):
        """Tests the function `find_video_load_luminance` in the file "hevc/unifiedloading.py".
        
        The test is successful if the saved image "hevc/pseudo_visualization/find_video_load_luminance.png"
        is a luminance frame of the video sequence "ice".
        
        """
        path_to_directory_file = 'hevc/pseudo_data/find_video_load_luminance/'
        prefix_filename = 'video_ice'
        
        luminance_uint8 = hevc.unifiedloading.find_video_load_luminance(path_to_directory_file,
                                                                        prefix_filename,
                                                                        idx_frame=19)
        tls.save_image('hevc/pseudo_visualization/find_video_load_luminance.png',
                       numpy.squeeze(luminance_uint8, axis=2))
    
    def test_predict_series_via_hevc_best_mode(self):
        """Tests the function `predict_series_via_hevc_best_mode` in the file "hevc/intraprediction/intraprediction.py".
        
        A 1st image is saved at
        "hevc/pseudo_visualization/predict_series_via_hevc_best_mode/prediction_hevc_best_mode_0.png".
        A 2nd image is saved at
        "hevc/pseudo_visualization/predict_series_via_hevc_best_mode/prediction_hevc_best_mode_1.png".
        The test is successful if, for each image, the
        direction of the best HEVC intra prediction mode
        in terms of PSNR is consistent with the prediction
        of the target patch displayed in this image.
        
        """
        height_intra_pattern = 49
        width_intra_pattern = 65
        width_target = 32
        
        # `index_channel` is the index of a YCbCr image channel.
        index_channel = 0
        
        rgb_0_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_cliff.jpg',
                                          'RGB')
        channel_0_uint8 = tls.rgb_to_ycbcr(rgb_0_uint8)[:, :, index_channel:index_channel + 1]
        rgb_1_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_library.jpg',
                                          'RGB')
        channel_1_uint8 = tls.rgb_to_ycbcr(rgb_1_uint8)[:, :, index_channel:index_channel + 1]
        channels_uint8 = numpy.stack((channel_0_uint8, channel_1_uint8),
                                     axis=0)
        intra_patterns_uint8 = channels_uint8[:, 220:220 + height_intra_pattern, 40:40 + width_intra_pattern, :].copy()
        targets_uint8 = intra_patterns_uint8[:, 1:1 + width_target, 1:1 + width_target, :].copy()
        intra_patterns_uint8[:, 1:, 1:, :] = 255*numpy.ones((2, height_intra_pattern - 1, width_intra_pattern - 1, 1),
                                                            dtype=numpy.uint8)
        (indices_hevc_best_mode, psnrs_hevc_best_mode, predictions_hevc_best_mode_uint8) = \
            hevc.intraprediction.intraprediction.predict_series_via_hevc_best_mode(intra_patterns_uint8,
                                                                                   targets_uint8)
        for i in range(intra_patterns_uint8.shape[0]):
            tls.save_image('hevc/pseudo_visualization/predict_series_via_hevc_best_mode/intra_pattern_{}.png'.format(i),
                           numpy.squeeze(intra_patterns_uint8[i, :, :, :], axis=2))
            tls.save_image('hevc/pseudo_visualization/predict_series_via_hevc_best_mode/target_patch_{}.png'.format(i),
                           numpy.squeeze(targets_uint8[i, :, :, :], axis=2))
            tls.save_image('hevc/pseudo_visualization/predict_series_via_hevc_best_mode/prediction_hevc_best_mode_{}.png'.format(i),
                           numpy.squeeze(predictions_hevc_best_mode_uint8[i, :, :, :], axis=2))
        print('Reminder about several HEVC intra prediction modes: planar 0, DC 1, horizontal (left->right) 10, vertical (top->bottom) 25.')
        print('\n1st case:')
        print('Index of the best HEVC intra prediction mode in terms of prediction PSNR: {}'.format(indices_hevc_best_mode[0]))
        print('PSNR between the target patch and its prediction via the best HEVC intra prediction mode: {}'.format(psnrs_hevc_best_mode[0]))
        print('\n2nd case:')
        print('Index of the best HEVC intra prediction mode in terms of prediction PSNR: {}'.format(indices_hevc_best_mode[1]))
        print('PSNR between the target patch and its prediction via the best HEVC intra prediction mode: {}'.format(psnrs_hevc_best_mode[1]))
    
    def test_predict_via_hevc_best_mode(self):
        """Tests the function `predict_via_hevc_best_mode` in the file "hevc/intraprediction/intraprediction.py".
        
        An image is saved at
        "hevc/pseudo_visualization/predict_via_hevc_best_mode/prediction_hevc_best_mode.png".
        The test is successful if the direction of the
        best HEVC intra prediction mode in terms of prediction
        PSNR is consistent with the prediction of the target
        patch displayed in this image.
        
        """
        height_intra_pattern = 49
        width_intra_pattern = 65
        width_target = 32
        index_channel = 0
        
        rgb_uint8 = tls.read_image_mode('hevc/pseudo_data/rgb_cliff.jpg',
                                        'RGB')
        channel_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, index_channel:index_channel + 1]
        intra_pattern_uint8 = channel_uint8[220:220 + height_intra_pattern, 40:40 + width_intra_pattern, :].copy()
        target_uint8 = intra_pattern_uint8[1:1 + width_target, 1:1 + width_target, :].copy()
        
        # The useless portion of the intra pattern is set to 255.
        intra_pattern_uint8[1:, 1:, :] = 255*numpy.ones((height_intra_pattern - 1, width_intra_pattern - 1, 1), dtype=numpy.uint8)
        (index_hevc_best_mode, psnr_hevc_best_mode, prediction_hevc_best_mode_uint8) = \
            hevc.intraprediction.intraprediction.predict_via_hevc_best_mode(intra_pattern_uint8,
                                                                            target_uint8)
        tls.save_image('hevc/pseudo_visualization/predict_via_hevc_best_mode/intra_pattern.png',
                       numpy.squeeze(intra_pattern_uint8, axis=2))
        tls.save_image('hevc/pseudo_visualization/predict_via_hevc_best_mode/target_patch.png',
                       numpy.squeeze(target_uint8, axis=2))
        tls.save_image('hevc/pseudo_visualization/predict_via_hevc_best_mode/prediction_hevc_best_mode.png',
                       numpy.squeeze(prediction_hevc_best_mode_uint8, axis=2))
        print('Reminder about several HEVC intra prediction modes: planar 0, DC 1, horizontal (left->right) 10, vertical (top->bottom) 25.')
        print('Index of the best HEVC intra prediction mode in terms of prediction PSNR: {}'.format(index_hevc_best_mode))
        print('PSNR between the target patch and its prediction via the best HEVC intra prediction mode in terms of prediction PSNR: {}'.format(psnr_hevc_best_mode))
    
    def test_predict_via_hevc_mode(self):
        """Tests the function `predict_via_hevc_mode` in the file "hevc/intraprediction/interface.pyx".
        
        The test is successful if, for each HEVC intra
        prediction mode, the prediction of the target patch
        computed by the function is identical to the prediction
        of the target patch computed by hand.
        
        """
        height_intra_pattern = 6
        width_intra_pattern = 5
        width_target = 4
        index_mode_0 = 10
        index_mode_1 = 18
        index_mode_2 = 25
        
        intra_pattern_uint8 = 255*numpy.ones((height_intra_pattern, width_intra_pattern, 1),
                                             dtype=numpy.uint8)
        intra_pattern_uint8[0, :, 0] = 1
        intra_pattern_uint8[:, 0, 0] = numpy.linspace(1,
                                                      height_intra_pattern,
                                                      num=height_intra_pattern,
                                                      dtype=numpy.uint8)
        prediction_0_uint8 = hevc.intraprediction.interface.predict_via_hevc_mode(intra_pattern_uint8,
                                                                                  width_target,
                                                                                  index_mode_0)
        prediction_1_uint8 = hevc.intraprediction.interface.predict_via_hevc_mode(intra_pattern_uint8,
                                                                                  width_target,
                                                                                  index_mode_1)
        prediction_2_uint8 = hevc.intraprediction.interface.predict_via_hevc_mode(intra_pattern_uint8,
                                                                                  width_target,
                                                                                  index_mode_2)
        print('HEVC intra prediction pattern of the target patch:')
        print(numpy.squeeze(intra_pattern_uint8, axis=2))
        print('Prediction of the target patch computed by the function via the HEVC intra prediction mode of index {}:'.format(index_mode_0))
        print(numpy.squeeze(prediction_0_uint8, axis=2))
        print('Prediction of the target patch computed by hand via the HEVC intra prediction mode of index {}:'.format(index_mode_0))
        print(numpy.array([[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]], dtype=numpy.uint8))
        print('Prediction of the target patch computed by the function via the HEVC intra prediction mode of index {}:'.format(index_mode_1))
        print(numpy.squeeze(prediction_1_uint8, axis=2))
        print('Prediction of the target patch computed by hand via the HEVC intra prediction mode of index {}:'.format(index_mode_1))
        print(numpy.array([[1, 1, 1, 1], [2, 1, 1, 1], [3, 2, 1, 1], [4, 3, 2, 1]], dtype=numpy.uint8))
        print('Prediction of the target patch computed by the function via the HEVC intra prediction mode of index {}:'.format(index_mode_2))
        print(numpy.squeeze(prediction_2_uint8, axis=2))
        print('Prediction of the target patch computed by hand via the HEVC intra prediction mode of index {}:'.format(index_mode_2))
        print(numpy.ones((width_target, width_target), dtype=numpy.uint8))
    
    def test_read_400_or_420(self):
        """Tests the function `read_400_or_420` in the file "hevc/running.py".
        
        For i = 0 ... 5, an image is saved at
        "hevc/pseudo_visualization/read_400_or_420/frame_i.png".
        The test is successful if these images
        form a coherent RGB video.
        
        """
        height_video = 288
        width_video = 352
        nb_frames = 240
        data_type = numpy.uint8
        path_to_video = 'hevc/pseudo_data/read_400_or_420/video_ice.yuv'
        
        video_uint8 = hevc.running.read_400_or_420(height_video,
                                                   width_video,
                                                   nb_frames,
                                                   data_type,
                                                   False,
                                                   path_to_video)
        numpy.save('hevc/pseudo_data/read_400_or_420/video_ice.npy',
                   video_uint8)
        for i in range(10):
            rgb_uint8 = tls.ycbcr_to_rgb(video_uint8[:, :, :, i])
            tls.save_image('hevc/pseudo_visualization/read_400_or_420/frame_{}.png'.format(i),
                           rgb_uint8)
    
    def test_search_1st_match_beacons(self):
        """Tests the function `search_1st_match_beacons` in the file "hevc/stats.py".
        
        The test is successful if, the 2nd row of the 3rd
        container in the 1st group of containers is filled.
        Besides, the 2nd container in the 2nd group of containers
        must be filled. The other containers must be unchanged.
        
        """
        line_text_0 = 'Rajon Rondo, Paul Pierce, beacon_0 1 beacon_1 -3 -2 -1'
        line_text_1 = 'Tyson Chandler bea_1 1 2'
        pairs_beacons = (
            ('beac_0', 'beac_1'),
            ('beacon_0', 'beac_1'),
            ('beacon_0', 'beacon_1')
        )
        beacons_before_integers = ('bea_0', 'bea_1')
        
        containers_0_int64 = numpy.zeros((len(pairs_beacons), 2, 3), dtype=numpy.int64)
        containers_1_int64 = numpy.zeros((len(beacons_before_integers), 2), dtype=numpy.int64)
        print('1st group of containers before they are filled by `search_1st_match_beacons`:')
        for i in range(containers_0_int64.shape[0]):
            print(containers_0_int64[i, :, :])
        print('2nd group of containers before they are filled by `search_1st_match_beacons`:')
        for i in range(containers_1_int64.shape[0]):
            print(containers_1_int64[i, :])
        hevc.stats.search_1st_match_beacons(line_text_0,
                                            pairs_beacons,
                                            beacons_before_integers,
                                            containers_0_int64,
                                            containers_1_int64)
        hevc.stats.search_1st_match_beacons(line_text_1,
                                            pairs_beacons,
                                            beacons_before_integers,
                                            containers_0_int64,
                                            containers_1_int64)
        print('1st group of containers after they are filled by `search_1st_match_beacons`:')
        for i in range(containers_0_int64.shape[0]):
            print(containers_0_int64[i, :, :])
        print('2nd group of containers after they are filled by `search_1st_match_beacons`:')
        for i in range(containers_1_int64.shape[0]):
            print(containers_1_int64[i, :])
    
    def test_write_400_or_420(self):
        """Tests the function `write_400_or_420` in the file "hevc/running.py".
        
        The test is successful if the original YCrCb video
        and its copy are identical.
        
        """
        path_to_original = 'hevc/pseudo_data/read_400_or_420/video_ice.yuv'
        path_to_copy = 'hevc/pseudo_data/read_400_or_420/video_ice_copy.yuv'
        
        video_uint8 = numpy.load('hevc/pseudo_data/read_400_or_420/video_ice.npy')
        
        # `hevc.running.write_400_or_420` raises a `IOError`
        # exception if a file already exists at `path_to_copy`.
        if os.path.isfile(path_to_copy):
            os.remove(path_to_copy)
        hevc.running.write_400_or_420(video_uint8,
                                      path_to_copy)
        if filecmp.cmp(path_to_original, path_to_copy):
            print('The original YCbCr video and its copy are identical.')
        else:
            print('The original YCbCr video and its copy are different.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests all the libraries in the directory "hevc".')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterHEVC()
    getattr(tester, 'test_' + args.name)()


