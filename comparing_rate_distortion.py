"""A script to compare HEVC with the substitution and HEVC with the switch in terms of rate-distortion.

The first argument of this script is `type_data`, the type
of data used to compare HEVC with the substitution and HEVC
with the switch in terms of rate-distortion. `type_data` can
take four values.
    "ycbcr": a YUV video sequence is loaded in 4:2:0 and the luminance
             channel of its first frame is extracted. The comparison is
             done on this luminance channel.
    "rgb": a RGB image is loaded and converted into luminance. The comparison
           is done on this luminance image.
    "kodak": the Kodak test set is loaded. The comparison is done on all
             the luminance images in the Kodak test set.
    "gray": a grayscale image is loaded. The comparison is done on this
            grayscale image.
The second argument of this script is `path_to_directory_data`, the path
to the directory containing either the file ".yuv", the RGB image or the
grayscale image.
The third argument of this script is `prefix_filename`, the prefix of the
name of either the file ".yuv", the RGB image or the grayscale image.

Note that the only case where the comparison is not done on
luminance is when `type_data` is equal to "gray".

Note also that, if `type_data` is equal to either "ycbcr", "rgb" or "gray",
`path_to_directory_data` and `prefix_filename` have to be provided.

"""

import argparse
import numpy
import os
import pickle

import hevc.constants
import hevc.performance
import hevc.unifiedloading
import tools.tools as tls

def compare_substitution_switch(luminances_before_encoding_hevc_uint8, path_to_cfg, qps_int, dict_substitution_switch,
                                dict_visualization_common, tuple_tags, path_to_directory_temp, path_to_directory_vis):
    """Compares HEVC with the substitution and HEVC with the switch in terms of rate-distortion.
    
    Parameters
    ----------
    luminances_before_encoding_hevc_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images before the encoding via HEVC. `luminances_before_encoding_hevc_uint8[i, :, :, :]`
        is the luminance image of index i. `luminances_before_encoding_hevc_uint8.shape[3]`
        is equal to 1.
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    qps_int : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        Quantization parameters.
    dict_substitution_switch : dict
        path_to_additional_directory : str
            Path to the directory containing the file "loading.py".
        path_to_mean_training : str
            Path to the file containing the mean pixel luminance
            computed over different luminance images. The path ends
            with ".pkl".
    dict_visualization_common : dict
        See the documentation of `compute_or_load_rates_psnrs`
        for further details.
    tuple_tags : tuple
        See the documentation of `compute_or_load_rates_psnrs`
        for further details.
    path_to_directory_temp : str
        Path to the directory whose subdirectories store the
        temporary files related to HEVC.
    path_to_directory_vis : str
        Path to the directory whose subdirectories store all the
        rates and PSNRs, plus visualizations.
    
    """
    (rates_regular, psnrs_regular) = compute_or_load_rates_psnrs(luminances_before_encoding_hevc_uint8,
                                                                 path_to_cfg,
                                                                 hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                                                 hevc.constants.PATH_TO_EXE_DECODER_REGULAR,
                                                                 qps_int,
                                                                 None,
                                                                 dict_visualization_common,
                                                                 tuple_tags,
                                                                 os.path.join(path_to_directory_temp, 'regular'),
                                                                 os.path.join(path_to_directory_vis, 'regular'))
    
    # Computation time is saved by computing one time
    # `rates_regular` and `psnrs_regular`, then reusing
    # them for drawing the rate-distortion curves when
    # `tag_pair` is equal to "single" and those when `tag_pair`
    # is equal to "pair".
    # As a reminder, "single" means that the PNNs inside HEVC
    # were trained on contexts without quantization noise and
    # "pair" means that, for large QPs, the PNNs inside HEVC
    # were trained on contexts with quantization noise.
    for tag_pair in ('single', 'pair'):
        dict_substitution_switch['path_to_file_paths_to_graphs_output'] = os.path.join('hevc/hm_common/paths_to_graphs_output',
                                                                                       '{}.txt'.format(tag_pair))
        (rates_substitution, psnrs_substitution) = compute_or_load_rates_psnrs(luminances_before_encoding_hevc_uint8,
                                                                               path_to_cfg,
                                                                               hevc.constants.PATH_TO_EXE_ENCODER_SUBSTITUTION,
                                                                               hevc.constants.PATH_TO_EXE_DECODER_SUBSTITUTION,
                                                                               qps_int,
                                                                               dict_substitution_switch,
                                                                               dict_visualization_common,
                                                                               tuple_tags,
                                                                               os.path.join(path_to_directory_temp, tag_pair, 'substitution'),
                                                                               os.path.join(path_to_directory_vis, tag_pair, 'substitution'))
        (rates_switch, psnrs_switch) = compute_or_load_rates_psnrs(luminances_before_encoding_hevc_uint8,
                                                                   path_to_cfg,
                                                                   hevc.constants.PATH_TO_EXE_ENCODER_SWITCH,
                                                                   hevc.constants.PATH_TO_EXE_DECODER_SWITCH,
                                                                   qps_int,
                                                                   dict_substitution_switch,
                                                                   dict_visualization_common,
                                                                   tuple_tags,
                                                                   os.path.join(path_to_directory_temp, tag_pair, 'switch'),
                                                                   os.path.join(path_to_directory_vis, tag_pair, 'switch'))
        
        # For computing Bjontegaard's metric, the regular HEVC
        # is always the reference.
        plot_rate_distortion_low_high_full(rates_substitution,
                                           psnrs_substitution,
                                           rates_regular,
                                           psnrs_regular,
                                           tuple_tags,
                                           ['HEVC substitution', 'HEVC regular'],
                                           os.path.join(path_to_directory_vis, tag_pair, 'curves', 'regular_substitution'))
        plot_rate_distortion_low_high_full(rates_switch,
                                           psnrs_switch,
                                           rates_regular,
                                           psnrs_regular,
                                           tuple_tags,
                                           ['HEVC switch', 'HEVC regular'],
                                           os.path.join(path_to_directory_vis, tag_pair, 'curves', 'regular_switch'))

def compute_or_load_rates_psnrs(luminances_before_encoding_hevc_uint8, path_to_cfg, path_to_exe_encoder,
                                path_to_exe_decoder, qps_int, dict_substitution_switch, dict_visualization_common,
                                tuple_tags, path_to_directory_temp, path_to_directory_rate_distortion):
    """Computes the rate and the PSNR of the encoding and decoding of each luminance image via HEVC for each quantization parameter or loads them.
    
    Parameters
    ----------
    luminances_before_encoding_hevc_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images before the encoding via HEVC. `luminances_before_encoding_hevc_uint8[i, :, :, :]`
        is the luminance image of index i. `luminances_before_encoding_hevc_uint8.shape[3]`
        is equal to 1.
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    path_to_exe_encoder : str
        Path to the executable of the HEVC encoder.
    path_to_exe_decoder : str
        Path to the executable of the HEVC decoder.
    qps_int : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        Quantization parameters.
    dict_substitution_switch : either dict or None
        path_to_additional_directory : str
            Path to the directory containing the file "loading.py".
        path_to_mean_training : str
            Path to the file containing the mean pixel luminance
            computed over different luminance images. The path ends
            with ".pkl".
        path_to_file_paths_to_graphs_output : str
            Path to the file containing the path to the output graph
            of each PNN used inside HEVC. The path ends with ".txt".
        If `dict_substitution_switch` is None, `path_to_exe_encoder`
        is the path to neither to the encoder of HEVC with the substitution
        nor the encoder of HEVC with the switch.
    dict_visualization_common : dict
        rows_columns_top_left : numpy.ndarray
            2D array whose data-type is smaller than
            `numpy.integer` in type hierarchy.
            `rows_columns_top_left[:, i]` contains the
            row and the column of the image pixel at the
            top-left of the crop of index i.
        width_crop : int
            Width of the crop.
        list_indices_rotation : list
            Each integer in this list is the index of a
            rotated luminance image.
    tuple_tags : tuple
        str
            Prefix of the name of the group `luminances_before_encoding_hevc_uint8`.
        list
            The string of index i in this list identifies
            the luminance image of index i within the group
            `luminances_before_encoding_hevc_uint8`.
    path_to_directory_temp : str
        Path to the directory storing the temporary files related to HEVC.
    path_to_directory_rate_distortion : str
        Path to the directory whose subdirectories store all the
        rates and PSNRs, plus visualizations.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this array
            is the rate of the encoding and decoding of the
            luminance image of index j via HEVC at the quantization
            parameter of index i.
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this array
            is the PSNR of the encoding and decoding of the 
            luminance image of index j via HEVC at the quantization
            parameter of index i.
    
    """
    # If the luminance images do not belong to a group of luminance
    # images, the identifier of the 1st luminance image is used to
    # label the rates and the PSNRs below.
    if tuple_tags[0]:
        tag_group = tuple_tags[0]
    else:
        tag_group = tuple_tags[1][0]
    
    # If the directory containing the rate-distortion
    # performance does not exist, it is created.
    if not os.path.isdir(path_to_directory_rate_distortion):
        os.makedirs(path_to_directory_rate_distortion)
    path_to_rates = os.path.join(path_to_directory_rate_distortion,
                                 'rates_{}.npy'.format(tag_group))
    path_to_psnrs = os.path.join(path_to_directory_rate_distortion,
                                 'psnrs_{}.npy'.format(tag_group))
    path_to_computation_times = os.path.join(path_to_directory_rate_distortion,
                                             'computation_times_{}.npy'.format(tag_group))
    if os.path.isfile(path_to_rates) and os.path.isfile(path_to_psnrs) and os.path.isfile(path_to_computation_times):
        rates = numpy.load(path_to_rates)
        psnrs = numpy.load(path_to_psnrs)
        print('The files at respectively "{0}", "{1}" and "{2}" already exist.'.format(path_to_rates,
                                                                                       path_to_psnrs,
                                                                                       path_to_computation_times))
    else:
        dict_visualization = dict_visualization_common.copy()
        dict_visualization['paths_to_saved_rotated_images_crops'] = \
            create_paths_to_saved_rotated_images_crops(qps_int,
                                                       tuple_tags,
                                                       path_to_directory_rate_distortion)
        if not os.path.isdir(path_to_directory_temp):
            os.makedirs(path_to_directory_temp)
        (rates, psnrs_expanded, computation_times) = \
            hevc.performance.compute_rates_psnrs(luminances_before_encoding_hevc_uint8,
                                                 os.path.join(path_to_directory_temp, 'luminance_before_encoding_hevc_{}.yuv'.format(tag_group)),
                                                 os.path.join(path_to_directory_temp, 'luminance_after_encoding_hevc_{}.yuv'.format(tag_group)),
                                                 os.path.join(path_to_directory_temp, 'luminance_after_decoding_hevc_{}.yuv'.format(tag_group)),
                                                 path_to_cfg,
                                                 os.path.join(path_to_directory_temp, 'bitstream_{}.bin'.format(tag_group)),
                                                 path_to_exe_encoder,
                                                 path_to_exe_decoder,
                                                 os.path.join(path_to_directory_temp, 'log_encoder_{}.txt'.format(tag_group)),
                                                 os.path.join(path_to_directory_temp, 'log_decoder_{}.txt'.format(tag_group)),
                                                 qps_int,
                                                 dict_substitution_switch,
                                                 dict_visualization)
        
        # `psnrs_expanded.shape[2]` is equal to 1 as a luminance image
        # has a single channel.
        psnrs = numpy.squeeze(psnrs_expanded,
                              axis=2)
        numpy.save(path_to_rates,
                   rates)
        numpy.save(path_to_psnrs,
                   psnrs)
        numpy.save(path_to_computation_times,
                   computation_times)
    return (rates, psnrs)

def create_paths_to_saved_rotated_images_crops(qps_int, tuple_tags, path_to_directory_root):
    """Creates a list of lists of lists of lists of paths to saved visualizations.
    
    Parameters
    ----------
    qps_int : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        Quantization parameters.
    tuple_tags : tuple
        str
            Prefix of the name of the group of luminance images.
        list
            The string of index i in this list identifies the
            luminance image of index i within the group of luminance
            images.
    path_to_directory_root : str
        Path to the directory whose subdirectories contains the
        saved visualizations.
    
    Returns
    -------
    list
        List of lists of lists of lists of paths. `paths_to_saved_rotated_images_crops[i][j]`
        contains the paths to saved visualizations for the
        reconstructed luminance image of index j after the
        decoding via HEVC at the quantization parameter of
        index i.
    
    """
    paths_to_saved_rotated_images_crops = []
    for qp in qps_int:
        paths_temp_qps = []
        path_to_directory_tag = os.path.join(path_to_directory_root,
                                             'qp_{}'.format(qp),
                                             tuple_tags[0])
        
        # If the directory containing the saved luminance images
        # does not exist, it is created.
        if not os.path.isdir(path_to_directory_tag):
            os.makedirs(path_to_directory_tag)
        
        # `len(tuple_tags[1])` is equal to the number luminance images.
        for i in range(len(tuple_tags[1])):
            paths_temp_qps.append(
                [
                    [os.path.join(path_to_directory_tag, '{}_reconstruction.png'.format(tuple_tags[1][i]))],
                    [os.path.join(path_to_directory_tag, '{}_crop0.png'.format(tuple_tags[1][i]))],
                    [os.path.join(path_to_directory_tag, '{}_crop1.png'.format(tuple_tags[1][i]))]
                ]
            )
        paths_to_saved_rotated_images_crops.append(paths_temp_qps)
    return paths_to_saved_rotated_images_crops

def plot_rate_distortion_low_high_full(rates_2d_0, psnrs_2d_0, rates_2d_1, psnrs_2d_1, tuple_tags,
                                       legend, path_to_directory_low_high_full):
    """Plots two rate-distortion curves and computes Bjontegaard's metric between the two curves for each luminance image.
    
    Three ranges of rates are considered: "low", "high", and "full"
    ranges. "full" includes all the quantization parameters. "high"
    includes the first half of the quantization parameters, i.e. relatively
    small quantization parameters, i.e. relatively large rates. "low"
    includes the second half of the quantization parameters, i.e. relatively
    large quantization parameters, i.e. relatively small rates.
    
    Parameters
    ----------
    rates_2d_0 : numpy.ndarray
        2D array.
        Rates for the 1st rate-distortion curve. `rates_2d_0[i, j]`
        is the rate of the point of index i in the 1st rate-distortion
        curve for the luminance image of index j.
    psnrs_2d_0 : numpy.ndarray
        2D array.
        PSNRs for the 1st rate-distortion curve. `psnrs_2d_0[i, j]`
        is the PSNR of the point of index i in the 1st rate-distortion
        curve for the luminance image of index j.
    rates_2d_1 : numpy.ndarray
        2D array.
        Rates for the 2nd rate-distortion curve. `rates_2d_1[i, j]`
        is the rate of the point of index i in the 2nd rate-distortion
        curve for the luminance image of index j.
    psnrs_2d_1 : numpy.ndarray
        2D array.
        PSNRs for the 2nd rate-distortion curve. `psnrs_2d_1[i, j]`
        is the PSNR of the point of index i in the 2nd rate-distortion
        curve for the luminance image of index j.
    tuple_tags : tuple
        str
            Prefix of the name of the group of luminance images.
        list
            The string of index i in this list identifies
            the luminance image of index i within the group
            of luminance images.
    legend : list
        Legend of the plot.
    path_to_directory_low_high_full : str
        Path to the directory storing the rate-distortion curves.
    
    Raises
    ------
    ValueError
        If `rates_2d_0.shape[0]` is not equal to `rates_2d_1.shape[0]`.
    
    """
    if rates_2d_0.shape[0] != rates_2d_1.shape[0]:
        raise ValueError('`rates_2d_0.shape[0]` is not equal to `rates_2d_1.shape[0]`.')
    
    # `rates_2d_0.shape[0]` is equal to the number of quantization
    # parameters.
    # `limit_low_high` is the index of the rate delimiting the
    # "low" and "high" ranges.
    limit_low_high = rates_2d_0.shape[0]//2
    path_to_directory_low = os.path.join(path_to_directory_low_high_full,
                                         'low',
                                         tuple_tags[0])
    if not os.path.isdir(path_to_directory_low):
        os.makedirs(path_to_directory_low)
    path_to_directory_high = os.path.join(path_to_directory_low_high_full,
                                          'high',
                                          tuple_tags[0])
    if not os.path.isdir(path_to_directory_high):
        os.makedirs(path_to_directory_high)
    path_to_directory_full = os.path.join(path_to_directory_low_high_full,
                                          'full',
                                          tuple_tags[0])
    if not os.path.isdir(path_to_directory_full):
        os.makedirs(path_to_directory_full)
    
    # `rates_2d_0.shape[1]` is equal to the number of luminance images.
    for i in range(rates_2d_0.shape[1]):
        tls.plot_two_rate_distortion_curves(rates_2d_0[:, i],
                                            psnrs_2d_0[:, i],
                                            rates_2d_1[:, i],
                                            psnrs_2d_1[:, i],
                                            legend,
                                            os.path.join(path_to_directory_full, '{}.png'.format(tuple_tags[1][i])))
        tls.plot_two_rate_distortion_curves(rates_2d_0[0:limit_low_high, i],
                                            psnrs_2d_0[0:limit_low_high, i],
                                            rates_2d_1[0:limit_low_high, i],
                                            psnrs_2d_1[0:limit_low_high, i],
                                            legend,
                                            os.path.join(path_to_directory_high, '{}.png'.format(tuple_tags[1][i])))
        tls.plot_two_rate_distortion_curves(rates_2d_0[limit_low_high:, i],
                                            psnrs_2d_0[limit_low_high:, i],
                                            rates_2d_1[limit_low_high:, i],
                                            psnrs_2d_1[limit_low_high:, i],
                                            legend,
                                            os.path.join(path_to_directory_low, '{}.png'.format(tuple_tags[1][i])))

def write_reference(luminances_before_encoding_hevc_uint8, dict_visualization_common, tuple_tags, path_to_directory_root):
    """Writes the luminance images to disk.
    
    Parameters
    ----------
    luminances_before_encoding_hevc_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images to be visualized. `luminances_before_encoding_hevc_uint8[i, :, :, :]`
        is the luminance image of index i. `luminances_before_encoding_hevc_uint8.shape[3]`
        is equal to 1.
    dict_visualization_common : dict
        rows_columns_top_left : numpy.ndarray
            2D array whose data-type is smaller than
            `numpy.integer` in type hierarchy.
            `rows_columns_top_left[:, i]` contains the
            row and the column of the image pixel at the
            top-left of the crop of index i.
        width_crop : int
            Width of the crop.
        list_indices_rotation : list
            Each integer in this list is the index of a
            rotated luminance image.
    tuple_tags : tuple
        str
            Prefix of the name of the group `luminances_before_encoding_hevc_uint8`.
        list
            The string of index i in this list identifies the
            luminance image of index i within the group `luminances_before_encoding_hevc_uint8`.
    path_to_directory_root : str
        Path to the directory whose subdirectories store
        the saved visualizations.
    
    """
    # If `tuple_tags[0]` is an empty string, `path_to_directory_root`
    # and `path_to_directory_tag` are two equivalent paths.
    path_to_directory_tag = os.path.join(path_to_directory_root,
                                         tuple_tags[0])
    if not os.path.isdir(path_to_directory_tag):
        os.makedirs(path_to_directory_tag)
    
    # `len(tuple_tags[1])` is equal to the number luminance images.
    for i in range(len(tuple_tags[1])):
        paths = [
            [os.path.join(path_to_directory_tag, '{}.png'.format(tuple_tags[1][i]))],
            [os.path.join(path_to_directory_tag, '{}_crop0.png'.format(tuple_tags[1][i]))],
            [os.path.join(path_to_directory_tag, '{}_crop1.png'.format(tuple_tags[1][i]))]
        ]
        tls.visualize_rotated_image(luminances_before_encoding_hevc_uint8[i, :, :, :],
                                    dict_visualization_common['rows_columns_top_left'],
                                    dict_visualization_common['width_crop'],
                                    i in dict_visualization_common['list_indices_rotation'],
                                    paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare HEVC with the substitution and HEVC with the switch in terms of rate-distortion.')
    parser.add_argument('type_data',
                        help='type of the data: either "kodak", "ycbcr", "rgb" or "gray"')
    parser.add_argument('--path_to_directory_data',
                        help='path to the directory containing either the file ".yuv", the RGB image or the grayscale image',
                        default='',
                        metavar='')
    parser.add_argument('--prefix_filename',
                        help='prefix of the name of either the file ".yuv", the RGB image or the grayscale image',
                        default='',
                        metavar='')
    args = parser.parse_args()
    
    path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
    
    # Two crops of each luminance image and its different reconstructions
    # are visualized as `dict_visualization_common['rows_columns_top_left'].shape[1]`
    # is equal to 2.
    dict_visualization_common = {
        'rows_columns_top_left': numpy.array([[100, 100], [200, 100]],
                                             dtype=numpy.int32),
        'width_crop': 80
    }
    
    # If `args.type_data` is equal to either "kodak", `luminances_before_encoding_hevc_uint8.shape[0]`
    # is equal to 24 because the Kodak test set contains 24 luminance images.
    if args.type_data == 'kodak':
        luminances_before_encoding_hevc_uint8 = numpy.load('sets/results/kodak/kodak.npy')[:, :, :, 0:1]
        tuple_tags = (
            'kodak',
            [str(i) for i in range(luminances_before_encoding_hevc_uint8.shape[0])]
        )
        with open('sets/results/kodak/list_indices_rotation.pkl', 'rb') as file:
            dict_visualization_common['list_indices_rotation'] = pickle.load(file)
    
    # If `args.type_data` is equal to either "ycbcr", "rgb" or "gray",
    # `luminances_before_encoding_hevc_uint8.shape[0]` is equal to 1
    # because a single luminance image is used for comparison.
    elif args.type_data in ('ycbcr', 'rgb', 'gray'):
        if args.type_data == 'ycbcr':
            loaded_uint8 = hevc.unifiedloading.find_video_load_luminance(args.path_to_directory_data,
                                                                         args.prefix_filename)
        elif args.type_data == 'rgb':
            loaded_uint8 = hevc.unifiedloading.find_rgb_load_luminance(args.path_to_directory_data,
                                                                       args.prefix_filename)
        else:
            loaded_uint8 = hevc.unifiedloading.find_load_grayscale(args.path_to_directory_data,
                                                                   args.prefix_filename)
        luminances_before_encoding_hevc_uint8 = numpy.expand_dims(loaded_uint8,
                                                                  0)
        tuple_tags = (
            '',
            [args.prefix_filename]
        )
        dict_visualization_common['list_indices_rotation'] = []
    else:
        raise ValueError('`args.type_data` is equal to neither "kodak" nor "ycbcr" nor "rgb" nor "gray".')
    qps_int = numpy.array([17, 19, 22, 24, 27, 32, 34, 37, 39, 42],
                          dtype=numpy.int32)
    dict_substitution_switch = {
        'path_to_additional_directory': 'hevc/hm_common/',
        'path_to_mean_training': 'sets/results/training_set/means/luminance/mean_training.pkl'
    }
    path_to_directory_temp = 'hevc/temp/comparing_rate_distortion/'
    path_to_directory_vis = 'hevc/visualization/rate_distortion/'
    
    # At this point, `dict_visualization_common` contains
    # the keys "rows_columns_top_left", "width_crop" and
    # "list_indices_rotation".
    write_reference(luminances_before_encoding_hevc_uint8,
                    dict_visualization_common,
                    tuple_tags,
                    os.path.join(path_to_directory_vis, 'reference'))
    compare_substitution_switch(luminances_before_encoding_hevc_uint8,
                                path_to_cfg,
                                qps_int,
                                dict_substitution_switch,
                                dict_visualization_common,
                                tuple_tags,
                                path_to_directory_temp,
                                path_to_directory_vis)
    print('The results are stored in the directory at \"{}\".'.format(path_to_directory_vis))


