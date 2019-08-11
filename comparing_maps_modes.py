"""A script to compare the maps of intra prediction modes of HEVC, those of HEVC with the substitution, and those of HEVC with the switch.

The first argument of this script is `type_data`, the type
of data used to compare the maps of intra prediction modes
of HEVC, those of HEVC with the substitution, and those of
HEVC with the switch. `type_data` can take two values.
    "ycbcr": a YUV video sequence is loaded in 4:2:0 and the luminance
             channel of its first frame is extracted. The comparison is
             done on this luminance channel.
    "rgb": a RGB image is loaded and converted into luminance. The comparison
           is done on this luminance image.
The second argument of this script is `path_to_directory_data`, the path
to the directory containing either the file ".yuv" or the RGB image.
The third argument of this script is `prefix_filename`, the prefix of the
name of either the file ".yuv" or the RGB image.

"""

import argparse
import numpy
import os

import hevc.constants
import hevc.running
import hevc.unifiedloading
import tools.tools as tls

def compare_maps_modes(luminance_before_encoding_hevc_uint8, path_to_cfg, qps_int, dict_substitution_switch,
                       prefix_filename, path_to_directory_temp, path_to_directory_vis):
    """Compares the maps of intra prediction modes of HEVC, those of HEVC with the substitution and those of HEVC with the switch for different quantization parameters.
    
    Parameters
    ----------
    luminance_before_encoding_hevc_uint8 : numpy.dnarray
        3D array with data-type `numpy.uint8`.
        Luminance image before the encoding via HEVC. `luminance_before_encoding_hevc_uint8.shape[2]`
        is equal to 1.
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    qps_int : numpy.ndarray
        1D array with data-type `numpy.int32`.
        Quantization parameters.
    dict_substitution_switch : dict
        path_to_additional_directory : str
            Path to the directory containing the file "loading.py".
        path_to_mean_training : str
            Path to the file containing the mean pixel luminance
            computed over different luminance images. The path ends
            with ".pkl".
        path_to_file_paths_to_graphs_output : str
            Path to the file containing the path to the output graph
            of each PNN used inside HEVC. The path ends with ".txt".
    prefix_filename : str
        Prefix of the name of the luminance image.
    path_to_directory_temp : str
        Path to the directory storing the temporary files related to HEVC.
    path_to_directory_vis : str
        Path to the directory storing the maps of intra prediction modes.
    
    """
    for i in range(qps_int.size):
        qp = qps_int[i].item()
        save_map_modes_if_not_exist(luminance_before_encoding_hevc_uint8,
                                    path_to_cfg,
                                    hevc.constants.PATH_TO_EXE_ENCODER_REGULAR,
                                    qp,
                                    None,
                                    prefix_filename,
                                    os.path.join(path_to_directory_temp, 'regular'),
                                    os.path.join(path_to_directory_vis, 'regular', 'qp_{}'.format(qp)))
        save_map_modes_if_not_exist(luminance_before_encoding_hevc_uint8,
                                    path_to_cfg,
                                    hevc.constants.PATH_TO_EXE_ENCODER_SUBSTITUTION,
                                    qp,
                                    dict_substitution_switch,
                                    prefix_filename,
                                    os.path.join(path_to_directory_temp, 'substitution'),
                                    os.path.join(path_to_directory_vis, 'substitution', 'qp_{}'.format(qp)))
        save_map_modes_if_not_exist(luminance_before_encoding_hevc_uint8,
                                    path_to_cfg,
                                    hevc.constants.PATH_TO_EXE_ENCODER_SWITCH,
                                    qp,
                                    dict_substitution_switch,
                                    prefix_filename,
                                    os.path.join(path_to_directory_temp, 'switch'),
                                    os.path.join(path_to_directory_vis, 'switch', 'qp_{}'.format(qp)))

def save_map_modes_if_not_exist(luminance_before_encoding_hevc_uint8, path_to_cfg, path_to_exe_encoder, qp,
                                dict_substitution_switch, prefix_filename, path_to_directory_temp, path_to_directory_vis):
    """Creates the two maps of intra prediction modes in HEVC if it does not already exist.
    
    Parameters
    ----------
    luminance_before_encoding_hevc_uint8 : numpy.dnarray
        3D array with data-type `numpy.uint8`.
        Luminance image before the encoding via HEVC. `luminance_before_encoding_hevc_uint8.shape[2]`
        is equal to 1.
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    path_to_exe_encoder : str
        Path to the executable of the HEVC encoder.
    qp : int
        Quantization parameter.
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
    prefix_filename : str
        Prefix of the name of the luminance image.
    path_to_directory_temp : str
        Path to the directory storing the temporary files related to HEVC.
    path_to_directory_vis : str
        Path to the directory storing the maps of intra prediction modes.
    
    """
    if not os.path.isdir(path_to_directory_vis):
        os.makedirs(path_to_directory_vis)
    path_to_thresholded_map_modes_luminance = os.path.join(path_to_directory_vis,
                                                           '{}_map_luminance.ppm'.format(prefix_filename))
    if os.path.isfile(path_to_thresholded_map_modes_luminance):
        print('The file at "{}" already exists.'.format(path_to_thresholded_map_modes_luminance))
    else:
        
        # The reconstructed luminance image is not needed.
        if not os.path.isdir(path_to_directory_temp):
            os.makedirs(path_to_directory_temp)
        hevc.running.encode_image(luminance_before_encoding_hevc_uint8,
                                  os.path.join(path_to_directory_temp, 'luminance_before_encoding_hevc_{}.yuv'.format(prefix_filename)),
                                  os.path.join(path_to_directory_temp, 'luminance_after_encoding_hevc_{}.yuv'.format(prefix_filename)),
                                  path_to_cfg,
                                  os.path.join(path_to_directory_temp, 'bitstream_{}.bin'.format(prefix_filename)),
                                  path_to_exe_encoder,
                                  qp,
                                  dict_substitution_switch,
                                  path_to_thresholded_map_modes_luminance=path_to_thresholded_map_modes_luminance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compares the maps of intra prediction modes of HEVC, those of HEVC with the substitution, and those of HEVC with the switch.')
    parser.add_argument('type_data',
                        help='type of the data: either "ycbcr" or "rgb"')
    parser.add_argument('path_to_directory_data',
                        help='path to the directory containing either the file ".yuv" or the RGB image')
    parser.add_argument('prefix_filename',
                        help='prefix of the name of either the file ".yuv" or the RGB image')
    args = parser.parse_args()
    
    # `luminance_before_encoding_hevc_uint8.shape[3]` is equal to 1
    # because the luminance image has a single channel.
    if args.type_data == 'ycbcr':
        luminance_before_encoding_hevc_uint8 = hevc.unifiedloading.find_video_load_luminance(args.path_to_directory_data,
                                                                                             args.prefix_filename)
    elif args.type_data == 'rgb':
        luminance_before_encoding_hevc_uint8 = hevc.unifiedloading.find_rgb_load_luminance(args.path_to_directory_data,
                                                                                           args.prefix_filename)
    else:
        raise ValueError('`args.type_data` is equal to neither "ycbcr" nor "rgb".')
    path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
    
    # For shorter experiments, QPs 27, 32, 37 can be removed
    # from `qps_int`. But, the lowest QP and the largest QP
    # should be kept as they enable to see whether there is
    # a link between the QP value and the frequency of selection
    # of the neural network mode.
    qps_int = numpy.array([22, 27, 32, 37, 42],
                          dtype=numpy.int32)
    dict_substitution_switch = {
        'path_to_additional_directory': 'hevc/hm_common/',
        'path_to_mean_training': 'sets/results/training_set/means/luminance/mean_training.pkl',
        
        # The PNN models that were trained using the training sets
        # containing contexts with HEVC compression artifacts are
        # not involved.
        'path_to_file_paths_to_graphs_output': 'hevc/hm_common/paths_to_graphs_output/single.txt'
    }
    path_to_directory_temp = 'hevc/temp/comparing_maps_modes/'
    path_to_directory_vis = 'hevc/visualization/map_intra_prediction_modes/'
    path_to_directory_vis_ref = os.path.join(path_to_directory_vis,
                                             'reference')
    if not os.path.isdir(path_to_directory_vis_ref):
        os.makedirs(path_to_directory_vis_ref)
    
    # The original luminance image is saved as, by comparing
    # the original luminance image and its maps of intra
    # prediction modes, it is possible to reveal a potential
    # link between the image texture and the selection of specific
    # intra prediction modes.
    tls.save_image(os.path.join(path_to_directory_vis_ref, '{}_luminance.png'.format(args.prefix_filename)),
                   numpy.squeeze(luminance_before_encoding_hevc_uint8, axis=2))
    compare_maps_modes(luminance_before_encoding_hevc_uint8,
                       path_to_cfg,
                       qps_int,
                       dict_substitution_switch,
                       args.prefix_filename,
                       path_to_directory_temp,
                       path_to_directory_vis)


