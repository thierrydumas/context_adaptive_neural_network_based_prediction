"""A script to collect statistics about the frequency of use of HEVC intra prediction modes.

The luminance channel of the YCbCr images in the BSDS test set
and the luminance channel of the YCbCr images in the INRIA
Holidays test set are used for the collection.

"""

import argparse
import numpy
import os
import time

import hevc.constants
import hevc.stats
import tools.tools as tls

def collect_statistics_bsds_holidays(luminances_bsds_uint8, luminances_holidays_uint8, path_to_before_encoding_hevc,
                                     path_to_after_encoding_hevc, path_to_cfg, path_to_bitstream, path_to_exe_encoder,
                                     qps_int, dict_substitution_switch, pairs_beacons, beacon_run, nb_modes, nb_widths_target,
                                     path_to_directory_stats):
    """Collects statistics about the frequency of use of HEVC intra prediction modes for several quantization parameters and plots them.
    
    The collection is done on the luminance channel of the YCbCr images
    in the BSDS test set and the luminance channel of the YCbCr images
    in the INRIA Holidays test set.
    
    Parameters
    ----------
    luminances_bsds_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images in the BSDS test set. `luminances_bsds_uint8[i, :, :, :]`
        is the luminance image of index i. `luminances_bsds_uint8.shape[3]`
        is equal to 1.
    luminances_holidays_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images in the INRIA Holidays set. `luminances_holidays_uint8[i, :, :, :]`
        is the luminance image of index i. `luminances_holidays_uint8.shape[3]`
        is equal to 1.
    path_to_before_encoding_hevc : str
        Path to the file storing a luminance image before the encoding
        via HEVC. `path_to_before_encoding_hevc` ends with ".yuv".
    path_to_after_encoding_hevc : str
        Path to the file storing the reconstructed luminance image
        after the encoding via HEVC. `path_to_after_encoding_hevc`
        ends with ".yuv".
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    path_to_bitstream : str
        Path to the bitstream file. `path_to_bitstream` ends
        with ".bin".
    path_to_exe_encoder : str
        Path to the executable of the HEVC encoder.
    qps_int : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        Quantization parameters.
    dict_substitution_switch : either dict or None
        See the documentation of `hevc.stats.encode_luminances_extract_statistics_from_files`
        for further details.
    pairs_beacons : tuple
        Each tuple in this tuple contains two strings.
        These two strings are two beacons for finding
        a specific series of statistics that depends on
        the HEVC intra prediction mode.
    beacon_run : str
        Beacon for finding the number of times the pipeline
        {fast selection, rate-distortion} selection is run
        for each width of target patch.
    nb_modes : int
        Number of intra prediction modes to be considered.
    nb_widths_target : int
        Number of widths of target patch to be considered.
    path_to_directory_stats : str
        Path to the directory whose subdirectories store the text
        files of statistics.
    
    """
    paths_to_directories_bsds_qp = []
    paths_to_directories_holidays_qp = []
    ratios_qps_bsds_float64 = numpy.zeros((qps_int.size, len(pairs_beacons), nb_modes, nb_widths_target))
    ratios_qps_holidays_float64 = numpy.zeros((qps_int.size, len(pairs_beacons), nb_modes, nb_widths_target))
    for i in range(qps_int.size):
        qp = qps_int[i].item()
        path_to_directory_bsds_qp = os.path.join(path_to_directory_stats,
                                                 'bsds',
                                                 'qp_{}'.format(qp))
        if not os.path.isdir(path_to_directory_bsds_qp):
            os.makedirs(path_to_directory_bsds_qp)
        path_to_directory_holidays_qp = os.path.join(path_to_directory_stats,
                                                     'holidays',
                                                     'qp_{}'.format(qp))
        if not os.path.isdir(path_to_directory_holidays_qp):
            os.makedirs(path_to_directory_holidays_qp)
        paths_to_stats_bsds = [
            os.path.join(path_to_directory_bsds_qp, 'stats_{}.txt'.format(j)) for j in range(luminances_bsds_uint8.shape[0])
        ]
        paths_to_stats_holidays = [
            os.path.join(path_to_directory_holidays_qp, 'stats_{}.txt'.format(j)) for j in range(luminances_holidays_uint8.shape[0])
        ]
        
        # For a given QP, the statistics files and the bar plots
        # are saved in the same directory.
        paths_to_directories_bsds_qp.append(path_to_directory_bsds_qp)
        paths_to_directories_holidays_qp.append(path_to_directory_holidays_qp)
        ratios_qps_bsds_float64[i, :, :, :] = \
            hevc.stats.encode_luminances_extract_statistics_from_files(luminances_bsds_uint8,
                                                                       path_to_before_encoding_hevc,
                                                                       path_to_after_encoding_hevc,
                                                                       path_to_cfg,
                                                                       path_to_bitstream,
                                                                       path_to_exe_encoder,
                                                                       qp,
                                                                       dict_substitution_switch,
                                                                       paths_to_stats_bsds,
                                                                       pairs_beacons,
                                                                       beacon_run,
                                                                       nb_modes,
                                                                       nb_widths_target)
        ratios_qps_holidays_float64[i, :, :, :] = \
            hevc.stats.encode_luminances_extract_statistics_from_files(luminances_holidays_uint8,
                                                                       path_to_before_encoding_hevc,
                                                                       path_to_after_encoding_hevc,
                                                                       path_to_cfg,
                                                                       path_to_bitstream,
                                                                       path_to_exe_encoder,
                                                                       qp,
                                                                       dict_substitution_switch,
                                                                       paths_to_stats_holidays,
                                                                       pairs_beacons,
                                                                       beacon_run,
                                                                       nb_modes,
                                                                       nb_widths_target)
        
    # For the moment, `ratios_qps_bsds_float64[:, 2, :, :]` and
    # `ratios_qps_holidays_float64[:, 2, :, :]` are not used.
    plot_bars_qps(ratios_qps_bsds_float64[:, 0:2, :, :],
                  ratios_qps_holidays_float64[:, 0:2, :, :],
                  paths_to_directories_bsds_qp,
                  paths_to_directories_holidays_qp)
    
def plot_bars_qps(ratios_qps_bsds_float64, ratios_qps_holidays_float64, paths_to_directories_bsds_qp,
                  paths_to_directories_holidays_qp):
    """Plots the frequency of use of HEVC intra prediction modes at different quantization parameters using bars.
    
    Parameters
    ----------
    ratios_qps_bsds_float64 : numpy.ndarray
        4D array with data-type `numpy.float64`.
        Frequencies depending on the HEVC intra prediction
        mode, computed using the luminance channel of the
        YCbCr images in the BSDS test set, for different QPs.
        `ratios_qps_bsds_float64.shape[1]` is equal to 2.
        The element at the position [i, j, k, l] in this array
        is associated to the QP of index i, the kind of frequency
        of index j, the mode of index k, and the width of target
        patch of index l in {4, 8, 16, 32, 64}.
    ratios_qps_holidays_float64 : numpy.ndarray
        4D array with data-type `numpy.float64`.
        Frequencies depending on the HEVC intra prediction
        mode, computed using the luminance channel of the
        YCbCr images in the INRIA Holidays test set, for different
        QPs. `ratios_qps_holidays_float64.shape[1]` is equal to 2.
        The element at the position [i, j, k, l] in this array
        is associated to the QP of index i, the kind of frequency
        of index j, the mode of index k, and the width of target
        patch of index l in {4, 8, 16, 32, 64}.
    paths_to_directories_bsds_qp : list
        `paths_to_directories_bsds_qp[i]` is the path to the directory
        storing the bar plots for BSDS set and the QP of index i.
    paths_to_directories_holidays_qp : list
        `paths_to_directories_holidays_qp[i]` is the path to the directory
        storing the bar plots for INRIA Holidays set and the QP of index i.
    
    """
    # `frequency_largest` gives a common maximum frequency
    # for all y-axis in the bar plots.
    frequency_largest = max(numpy.amax(ratios_qps_bsds_float64).item(),
                            numpy.amax(ratios_qps_holidays_float64).item())
    x_values_int32 = numpy.linspace(0,
                                    ratios_qps_bsds_float64.shape[2] - 1,
                                    num=ratios_qps_bsds_float64.shape[2],
                                    dtype=numpy.int32)
    for i in range(ratios_qps_bsds_float64.shape[0]):
        plot_bars_qp(x_values_int32,
                     ratios_qps_bsds_float64[i, :, :, :],
                     ratios_qps_holidays_float64[i, :, :, :],
                     frequency_largest,
                     paths_to_directories_bsds_qp[i],
                     paths_to_directories_holidays_qp[i])

def plot_bars_qp(x_values_int32, ratios_bsds_float64, ratios_holidays_float64, frequency_largest,
                 path_to_directory_bsds_qp, path_to_directory_holidays_qp):
    """Plots the frequency of use of HEVC intra prediction modes at a quantization parameter using bars.
    
    Parameters
    ----------
    x_values_int32 : numpy.ndarray
        1D array with data-type `numpy.int32`.
        Indices of the modes to be considered.
    ratios_bsds_float64 : numpy.ndarray
        3D array with data-type `numpy.float64`.
        Frequencies depending on the HEVC intra prediction
        mode, computed using the luminance channel of the
        YCbCr images in the BSDS test set. `ratios_bsds_float64.shape[0]`
        is equal to 2. The element at the position [i, j, k]
        in this array is associated to the kind of frequency
        of index i, the mode of index j, and the width of target
        patch of index k in {4, 8, 16, 32, 64}.
    ratios_holidays_float64 : numpy.ndarray
        3D array with data-type `numpy.float64`.
        Frequencies depending on the HEVC intra prediction
        mode, computed using the luminance channel of the
        YCbCr images in the INRIA Holidays test set. `ratios_holidays_float64.shape[0]`
        is equal to 2. The element at the position [i, j, k]
        in this array is associated to the kind of frequency
        of index i, the mode of index j, and the width of target
        patch of index k in {4, 8, 16, 32, 64}.
    frequency_largest : float
        Maximum frequency for all y-axis in the bar plots.
    path_to_directory_bsds_qp : str
        Path to the directory in which the bar plots for the
        luminance channel of the YCbCr images in the BSDS test
        set are saved.
    path_to_directory_holidays_qp : str
        Path to the directory in which the bar plots for the
        luminance channel of the YCbCr images in the INRIA Holidays
        test set are saved.
    
    Raises
    ------
    AssertionError
        If frequencies do not sum to 1.0 for the luminance channel
        of the YCbCr images in the BSDS test set.
    AssertionError
        If frequencies do not sum to 1.0 for the luminance channel
        of the YCbCr images in the INRIA Holidays test set.
    
    """
    numpy.testing.assert_almost_equal(numpy.sum(ratios_bsds_float64, axis=1),
                                      1.,
                                      decimal=9,
                                      err_msg='Frequencies do not sum to 1.0 for the luminance channel of the YCbCr images in the BSDS test set.')
    numpy.testing.assert_almost_equal(numpy.sum(ratios_holidays_float64, axis=1),
                                      1.,
                                      decimal=9,
                                      err_msg='Frequencies do not sum to 1.0 for the luminance channel of the YCbCr images in the INRIA Holidays test set.')
    tuple_yaxis_limits = (0., tls.ceil_float(frequency_largest, 1))
    for i in range(ratios_bsds_float64.shape[2]):
        
        # The precedence of ... + ... (addition) is higher
        # than the precedence of ... << ... (bitshift).
        str_width_target = str(1 << i + 2)
        tls.plot_bars_yaxis_limits(x_values_int32,
                                   ratios_bsds_float64[0, :, i],
                                   tuple_yaxis_limits,
                                   r'$\epsilon_{' + str_width_target + '}$ for each mode index',
                                   os.path.join(path_to_directory_bsds_qp, 'win_fast_selection_{}.png'.format(str_width_target)))
        tls.plot_bars_yaxis_limits(x_values_int32,
                                   ratios_holidays_float64[0, :, i],
                                   tuple_yaxis_limits,
                                   r'$\epsilon_{' + str_width_target + '}$ for each mode index',
                                   os.path.join(path_to_directory_holidays_qp, 'win_fast_selection_{}.png'.format(str_width_target)))
        tls.plot_bars_yaxis_limits(x_values_int32,
                                   ratios_bsds_float64[1, :, i],
                                   tuple_yaxis_limits,
                                   r'$\nu_{' + str_width_target + '}$ for each mode index',
                                   os.path.join(path_to_directory_bsds_qp, 'win_rate_distortion_selection_{}.png'.format(str_width_target)))
        tls.plot_bars_yaxis_limits(x_values_int32,
                                   ratios_holidays_float64[1, :, i],
                                   tuple_yaxis_limits,
                                   r'$\nu_{' + str_width_target + '}$ for each mode index',
                                   os.path.join(path_to_directory_holidays_qp, 'win_rate_distortion_selection_{}.png'.format(str_width_target)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects statistics about the frequency of use of HEVC intra prediction modes.')
    parser.add_argument('type_hevc',
                        help='HEVC type. It can be either "regular", "substitution" or "switch"')
    args = parser.parse_args()
    
    if args.type_hevc == 'regular':
        dict_substitution_switch = None
        path_to_exe_encoder = hevc.constants.PATH_TO_EXE_ENCODER_REGULAR
        nb_modes = 35
    elif args.type_hevc in ('substitution', 'switch'):
        dict_substitution_switch = {
            'path_to_additional_directory': 'hevc/hm_common/',
            'path_to_mean_training': 'sets/results/training_set/means/luminance/mean_training.pkl',
            
            # The PNN models that were trained using the training sets
            # containing contexts with HEVC compression artifacts are
            # not involved.
            'path_to_file_paths_to_graphs_output': 'hevc/hm_common/paths_to_graphs_output/single.txt'
        }
        if args.type_hevc == 'substitution':
            path_to_exe_encoder = hevc.constants.PATH_TO_EXE_ENCODER_SUBSTITUTION
            nb_modes = 35
        else:
            path_to_exe_encoder = hevc.constants.PATH_TO_EXE_ENCODER_SWITCH
            
            # For HEVC with the switch between the PNNs and the HEVC
            # intra prediction modes contains 36 modes.
            nb_modes = 36
    else:
        raise ValueError('The HEVC type is neither "regular" nor "substitution" nor "switch".')
    path_to_directory_temp = os.path.join('hevc/temp/collecting_stats_hevc_modes/',
                                          args.type_hevc)
    if not os.path.isdir(path_to_directory_temp):
        os.makedirs(path_to_directory_temp)
    path_to_before_encoding_hevc = os.path.join(path_to_directory_temp,
                                                'luminance_before_encoding_hevc.yuv')
    path_to_after_encoding_hevc = os.path.join(path_to_directory_temp,
                                               'luminance_after_encoding_hevc.yuv')
    path_to_cfg = 'hevc/configuration/intra_main_rext.cfg'
    path_to_bitstream = os.path.join(path_to_directory_temp,
                                     'bitstream.bin')
    qps_int = numpy.array([22, 27, 32, 37, 42],
                          dtype=numpy.int32)
    path_to_directory_stats = os.path.join('hevc/visualization/frequency_use_intra_prediction_modes/',
                                           args.type_hevc)
    pairs_beacons = (
        ('index', 'wins the fast selection:'),
        ('index', 'wins the rate-distortion selection:'),
        ('index', 'is found in the fast list:')
    )
    beacon_run = '{fast selection, rate-distortion selection} is run:'
    nb_widths_target = 5
    
    # The unused channels are not kept in RAM.
    luminances_bsds_uint8 = numpy.load('sets/results/bsds/bsds.npy')[:, :, :, 0:1]
    luminances_holidays_uint8 = numpy.load('sets/results/holidays/holidays.npy')[:, :, :, 0:1]
    
    # The collection of statistics about the frequency
    # of use of HEVC intra prediction modes involves many
    # images, thus taking long time.
    t_start = time.time()
    collect_statistics_bsds_holidays(luminances_bsds_uint8,
                                     luminances_holidays_uint8,
                                     path_to_before_encoding_hevc,
                                     path_to_after_encoding_hevc,
                                     path_to_cfg,
                                     path_to_bitstream,
                                     path_to_exe_encoder,
                                     qps_int,
                                     dict_substitution_switch,
                                     pairs_beacons,
                                     beacon_run,
                                     nb_modes,
                                     nb_widths_target,
                                     path_to_directory_stats)
    t_stop = time.time()
    nb_hours = int((t_stop - t_start)/3600)
    nb_minutes = int((t_stop - t_start)/60)
    print('\nThe collection of statistics about the frequency of use of intra prediction modes in HEVC "{0}" toke {1} hours and {2} minutes.'.format(args.type_hevc, nb_hours, nb_minutes - 60*nb_hours))


