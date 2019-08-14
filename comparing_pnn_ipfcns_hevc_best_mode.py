"""A script to compare the predictions provided by PNN, the predictions provided by an Intra
Prediction Fully-Connected Network (IPFCN-S), and the predictions provided by the best HEVC
intra prediction mode in terms of prediction PSNR.

For this comparison, the BSDS test set and the Kodak test set are used.

Note that, in the paper "Fully-connected network-based intra prediction
for image coding", written by Jiahao Li, Bin Li, Jizheng Xu, Ruiqin Xiong
and Wen Gao, there exist two versions of IPFCN-S. The 1st version, simply
called IPFCN-S, is more complex but provides predictions of higher accuracy.
The 2nd version, called IPFCN-S-L, is less complex but gives predictions of
lower accuracy. Here, IPFCN-S is used for comparison.

Caffe has to be built in CPU-only mode. Otherwise, Tensorflow
and Caffe share a GPU, which is not possible.

"""

import argparse
import numpy
import os
import pickle
import tensorflow as tf

import ipfcns.ipfcns
import hevc.intraprediction.intraprediction
import parsing.parsing
import pnn.batching
import pnn.PredictionNeuralNetwork
import pnn.visualization
import sets.common
import tools.tools as tls

def compute_performance_neural_network_vs_hevc_best_mode(targets_uint8, predictions_nn_uint8, psnrs_hevc_best_mode):
    """Compute the performance of the neural network predictor versus the best HEVC intra prediction mode in terms of prediction PSNR.
    
    Parameters
    ----------
    targets_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Target patches. `targets_uint8[i, :, :, :]` is the
        target patch of index i. `targets_uint8.shape[3]` is equal
        to 1.
    predictions_nn_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Prediction of the target patches via the neural
        network predictor. `predictions_nn_uint8[i, :, :, :]`
        is the prediction of the target patch of index i
        via the neural network predictor. `predictions_nn_uint8.shape[3]`
        is equal to 1.
    psnrs_hevc_best_mode : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Prediction PSNRs via the best HEVC intra prediction
        mode in terms of prediction PSNR. `psnrs_hevc_best_mode[i]`
        is the PSNR between the target patch of index i and
        its prediction via this mode.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Prediction PSNRs via the neural network predictor.
            Its element of index i is the PSNR between the target
            patch of index i and its prediction via the neural
            network predictor.
        float
            Frequency of the prediction PSNR of the neural network
            predictor being larger than the prediction PSNR of the best
            HEVC intra prediction mode in terms of prediction PSNR.
    
    """
    nb_targets = targets_uint8.shape[0]
    psnrs_nn = numpy.zeros(nb_targets)
    for i in range(nb_targets):
        squeezed_target_uint8 = numpy.squeeze(targets_uint8[i, :, :, :],
                                              axis=2)
        squeezed_prediction_nn_uint8 = numpy.squeeze(predictions_nn_uint8[i, :, :, :],
                                                     axis=2)
        psnrs_nn[i] = tls.compute_psnr(squeezed_target_uint8,
                                       squeezed_prediction_nn_uint8)
    frequency_win_nn = float(numpy.count_nonzero(psnrs_nn - psnrs_hevc_best_mode > 0.))/nb_targets
    return (psnrs_nn, frequency_win_nn)

def predict_mask(channels_uint8, width_target, row_1sts, col_1sts, batch_size, mean_training,
                 tuple_width_height_masks_val, sess, predictor, net_ipfcns, path_to_directory_vis):
    """Predicts each target patch from its masked context via PNN.
    
    Each target patch is also predicted from its intra pattern via
    the best HEVC intra prediction mode in terms of prediction PSNR.
    Each target patch is also predicted from its reference lines via
    IPFCN-S.
    
    Parameters
    ----------
    channels_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. `channels_uint8[i, :, :, :]` is the
        channel of the image of index i. `channels_uint8.shape[3]`
        is equal to 1.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row of the 1st pixel of the
        context of index i in the image channel.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column of the 1st pixel of the
        context of index i in the image channel.
    batch_size : int
        Number of predictions PNN computes in parallel.
    mean_training : float
        Mean pixel intensity computed over the same channel
        of different YCbCr images.
    tuple_width_height_masks_val : tuple
        The 1st integer in this tuple is the
        width of the mask that covers the right
        side of the context portion located above
        the target patch. The 2nd integer in this
        tuple is the height of the mask that covers
        the bottom of the context portion located
        on the left side of the target patch. The
        two masks are used during the validation phase.
    sess : Session
        Session that runs the graph of PNN.
    predictor : PredictionNeuralNetwork
        PNN instance.
    net_ipfcns : either Net (class in Caffe) or None
        IPFCN-S instance.
    path_to_directory_vis : str
        Path to the directory in which the visualizations
        of the predictions are saved.
    
    Raises
    ------
    RuntimeError
        If the target patches have been altered.
    
    """
    tuple_batches_float32 = sets.common.extract_context_portions_targets_from_channels_plus_preprocessing(channels_uint8,
                                                                                                          width_target,
                                                                                                          row_1sts,
                                                                                                          col_1sts,
                                                                                                          mean_training,
                                                                                                          tuple_width_height_masks_val,
                                                                                                          predictor.is_fully_connected)
    
    # In a given image channel, the difference between
    # the row of the 1st context pixel and the row of its
    # 1st intra pattern pixel is equal to `width_target - 1`
    # pixels. The difference between the column of the 1st
    # context pixel and the column of its 1st intra pattern
    # pixel is equal to `width_target - 1` pixels.
    row_refs = row_1sts + width_target - 1
    col_refs = col_1sts + width_target - 1
    intra_patterns_uint8 = hevc.intraprediction.intraprediction.extract_intra_patterns(channels_uint8,
                                                                                       width_target,
                                                                                       row_refs,
                                                                                       col_refs,
                                                                                       tuple_width_height_masks_val)
    predictions_pnn_float32 = pnn.batching.predict_by_batch_via_pnn(tuple_batches_float32[0:-1],
                                                                    sess,
                                                                    predictor,
                                                                    batch_size)
    
    # The batch of target patches is the last element of `tuple_batches_float32`.
    targets_float32 = tuple_batches_float32[-1]
    
    # `mean_training` was subtracted from the target
    # patches and, now, it is added to them. Therefore,
    # `targets_off_float32` should contain whole numbers
    # exclusively. This means that `tls.cast_float_to_uint8`
    # should not alter the target patches.
    targets_off_float32 = targets_float32 + mean_training
    if numpy.any(numpy.modf(targets_off_float32)[0]):
        raise RuntimeError('The target patches have been altered.')
    targets_uint8 = tls.cast_float_to_uint8(targets_off_float32)
    predictions_pnn_uint8 = tls.cast_float_to_uint8(predictions_pnn_float32 + mean_training)
    dictionary_performance = {}
    (dictionary_performance['indices_hevc_best_mode'], dictionary_performance['psnrs_hevc_best_mode'], predictions_hevc_best_mode_uint8) = \
        hevc.intraprediction.intraprediction.predict_series_via_hevc_best_mode(intra_patterns_uint8,
                                                                               targets_uint8)
    (dictionary_performance['psnrs_pnn'], dictionary_performance['frequency_win_pnn']) = \
        compute_performance_neural_network_vs_hevc_best_mode(targets_uint8,
                                                             predictions_pnn_uint8,
                                                             dictionary_performance['psnrs_hevc_best_mode'])
    tls.histogram(dictionary_performance['psnrs_pnn'] - dictionary_performance['psnrs_hevc_best_mode'],
                  'Difference in PSNRs',
                  os.path.join(path_to_directory_vis, 'difference_psnrs_pnn_hevc_best_mode.png'))
    if predictor.is_fully_connected:
        coefficient_enlargement = 4
    else:
        coefficient_enlargement = None
    for i in range(20):
        if predictor.is_fully_connected:
            pnn.visualization.visualize_flattened_context(tuple_batches_float32[0][i, :],
                                                          width_target,
                                                          mean_training,
                                                          os.path.join(path_to_directory_vis, 'masked_context_{}.png'.format(i)),
                                                          coefficient_enlargement=coefficient_enlargement)
        else:
            pnn.visualization.visualize_context_portions(tuple_batches_float32[0][i, :, :, :],
                                                         tuple_batches_float32[1][i, :, :, :],
                                                         mean_training,
                                                         os.path.join(path_to_directory_vis, 'masked_context_{}.png'.format(i)))
        tls.save_image(os.path.join(path_to_directory_vis, 'target_{}.png'.format(i)),
                       numpy.squeeze(targets_uint8[i, :, :, :], axis=2),
                       coefficient_enlargement=coefficient_enlargement)
        tls.save_image(os.path.join(path_to_directory_vis, 'intra_pattern_{}.png'.format(i)),
                       numpy.squeeze(intra_patterns_uint8[i, :, :, :], axis=2),
                       coefficient_enlargement=coefficient_enlargement)
        tls.save_image(os.path.join(path_to_directory_vis, 'prediction_pnn_{}.png'.format(i)),
                       numpy.squeeze(predictions_pnn_uint8[i, :, :, :], axis=2),
                       coefficient_enlargement=coefficient_enlargement)
        tls.save_image(os.path.join(path_to_directory_vis, 'prediction_hevc_best_mode_{}.png'.format(i)),
                       numpy.squeeze(predictions_hevc_best_mode_uint8[i, :, :, :], axis=2),
                       coefficient_enlargement=coefficient_enlargement)
    
    # IPFCN-S is used only if there is no masking during
    # the validation phase.
    if net_ipfcns is not None and tuple_width_height_masks_val == (0, 0):
        predict_without_mask_via_ipfcns(channels_uint8,
                                        width_target,
                                        row_1sts,
                                        col_1sts,
                                        batch_size,
                                        net_ipfcns,
                                        dictionary_performance,
                                        coefficient_enlargement,
                                        path_to_directory_vis)
    with open(os.path.join(path_to_directory_vis, 'dictionary_performance.pkl'), 'wb') as file:
        pickle.dump(dictionary_performance, file, protocol=2)

def predict_masks(channels_uint8, width_target, row_1sts, col_1sts, batch_size, mean_training,
                  tuples_width_height_masks_tr, tuples_width_height_masks_val, sess, predictor,
                  net_ipfcns_optional, path_to_directory_coeffs_load_save, path_to_directory_coeffs_vis):
    """Predicts each target patch from its masked context via PNNs, each trained using a different masking technique.
    
    Each target patch is also predicted from its intra pattern via
    the best HEVC intra prediction mode in terms of prediction PSNR.
    Each target patch is also predicted from its reference lines via
    IPFCN-S.
    
    Parameters
    ----------
    channels_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. `channels_uint8[i, :, :, :]` is the
        channel of the image of index i. `channels_uint8.shape[3]`
        is equal to 1.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row of the 1st pixel of the
        context of index i in the image channel.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column of the 1st pixel of the
        context of index i in the image channel.
    batch_size : int
        Number of predictions PNN computes in parallel.
    mean_training : float
        Mean pixel intensity computed over the same channel
        of different YCbCr images.
    tuples_width_height_masks_tr : tuple
        Each tuple in this tuple contains two integers.
        The 1st integer is the width of the mask that
        covers the right side of the context portion
        located above the target patch. The 2nd integer
        is the height of the mask that covers the bottom
        of the context portion located on the left side
        of the target patch. The two masks are used during
        the training phase.
    tuples_width_height_masks_val : tuple
        Each tuple in this tuple contains two integers.
        The 1st integer is the width of the mask that
        covers the right side of the context portion
        located above the target patch. The 2nd integer
        is the height of the mask that covers the bottom
        of the context portion located on the left side
        of the target patch. The two masks are used during
        the validation phase.
    sess : Session
        Session that runs the graph of PNN.
    predictor : PredictionNeuralNetwork
        PNN instance.
    net_ipfcns_optional : either Net (class in Caffe) or None
        IPFCN-S instance.
    path_to_directory_coeffs_load_save : str
        Path to the directory in which the PNN models
        with the same architecture and the same scaling
        coefficients in their objective function to be
        minimized are saved.
    path_to_directory_coeffs_vis : str
        Path to the directory in which the visualizations
        of the predictions are saved.
    
    """
    for tuple_width_height_masks_tr in tuples_width_height_masks_tr:
        
        # Only the PNN model trained via random context masking is
        # compared to the IPFCN-S.
        if tuple_width_height_masks_tr:
            net_ipfcns = None
        else:
            net_ipfcns = net_ipfcns_optional
        
        # The width of the mask that covers the right side
        # of the context portion located above the target
        # patch and the height of the mask that covers the
        # bottom of the context portion located on the left
        # side of the target patch are not necessarily the
        # same during the training and validation phases.
        if tuple_width_height_masks_tr:
            tag_masks_tr = 'masks_tr_{0}_{1}'.format(tuple_width_height_masks_tr[0],
                                                     tuple_width_height_masks_tr[1])
        else:
            tag_masks_tr = 'masks_tr_random'
        path_to_directory_load_save = os.path.join(path_to_directory_coeffs_load_save,
                                                   tag_masks_tr)
        
        # The loop iteration ends if the directory at `path_to_directory_load_save`
        # does not exist or it does not contain file ".ckpt".
        if not os.path.isdir(path_to_directory_load_save):
            continue
        iters_training = tls.collect_integer_between_tags_in_each_filename(path_to_directory_load_save,
                                                                           ('.ckpt.meta',),
                                                                           'model_',
                                                                           '.ckpt')
        if not iters_training:
            continue
        
        # The PNN model with the longest training time is restored.
        path_to_restore = os.path.join(path_to_directory_load_save,
                                       'model_{}.ckpt'.format(iters_training[-1]))
        predictor.initialization(sess,
                                 path_to_restore)
        for tuple_width_height_masks_val in tuples_width_height_masks_val:
            tag_masks_val = 'masks_val_{0}_{1}'.format(tuple_width_height_masks_val[0],
                                                       tuple_width_height_masks_val[1])
            path_to_directory_vis = os.path.join(path_to_directory_coeffs_vis,
                                                 tag_masks_tr,
                                                 tag_masks_val)
            
            # If the directory containing all the saved images does
            # not exist, it is created.
            if not os.path.isdir(path_to_directory_vis):
                os.makedirs(path_to_directory_vis)
            predict_mask(channels_uint8,
                         width_target,
                         row_1sts,
                         col_1sts,
                         batch_size,
                         mean_training,
                         tuple_width_height_masks_val,
                         sess,
                         predictor,
                         net_ipfcns,
                         path_to_directory_vis)

def predict_masks_kodak_bsds(channels_kodak_uint8, channels_bsds_uint8, width_target, index_channel, coeff_l2_norm_pred_error,
                             coeff_grad_error, is_fully_connected, is_pair, is_compared_ipfcns, batch_size):
    """Predicts each target patch from its masked context via PNNs, each trained using a different masking technique, on the Kodak and BSDS test sets.
    
    Parameters
    ----------
    channels_kodak_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Same channel of the YCbCr images in the Kodak test set.
        `channels_kodak_uint8[i, :, :, :]` is the channel of the
        YCbCr image of index i in the Kodak test set. `channels_kodak_uint8.shape[3]`
        is equal to 1.
    channels_bsds_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Same channel of the YCbCr images in the BSDS test set.
        `channels_bsds_uint8[i, :, :, :]` is the channel of the
        YCbCr image of index i in the BSDS test set. `channels_bsds_uint8.shape[3]`
        is equal to 1.
    width_target : int
        Width of the target patch.
    index_channel : int
        Channel index.
    coeff_l2_norm_pred_error : float
        Coefficient that scales the l2-norm prediction error w.r.t the
        gradient error and the l2-norm weight decay.
    coeff_grad_error : float
        Coefficient that scales the gradient error w.r.t the l2-norm
        prediction error and the l2-norm weight decay.
    is_fully_connected : bool
        Is PNN fully-connected?
    is_pair : bool
        Do the contexts in the training sets for training the different PNNs
        contain HEVC compression artifacts?
    is_compared_ipfcns : bool
        Is PNN compared with IPFCN-S?
    batch_size : int
        Number of predictions PNN computes in parallel.
    
    Raises
    ------
    ValueError
        If `index_channel` does not belong to {0, 1, 2}.
    
    """
    # The Numpy random seed is reset every time `predict_masks_kodak_bsds`
    # is called.
    numpy.random.seed(seed=0)
    tuples_width_height_masks_tr = (
        (),
        (0, 0),
        (0, width_target),
        (width_target, 0),
        (width_target, width_target)
    )
    tuples_width_height_masks_val = (
        (0, 0),
        (0, width_target),
        (width_target, 0),
        (width_target, width_target)
    )
    
    if is_fully_connected:
        tag_arch = 'fully_connected'
    else:
        tag_arch = 'convolutional'
    if is_pair:
        tag_pair = 'pair'
    else:
        tag_pair = 'single'
    if index_channel == 0:
        tag_channel = 'luminance'
    elif index_channel == 1:
        tag_channel = 'chrominance_blue'
    elif index_channel == 2:
        tag_channel = 'chrominance_red'
    else:
        raise ValueError('`index_channel` does not belong to {0, 1, 2}.')
    tag_coeffs = '{0}_{1}'.format(tls.float_to_str(coeff_l2_norm_pred_error),
                                  tls.float_to_str(coeff_grad_error))
    
    # `suffix_paths` enables to integrate the common characteristics
    # of PNNs into the path to the directory containing the PNN models
    # to be loaded and the path to the directory storing the results
    # of the current comparison.
    suffix_paths = os.path.join('width_target_{}'.format(width_target),
                                tag_arch,
                                tag_pair,
                                tag_channel,
                                tag_coeffs)
    path_to_directory_coeffs_load_save = os.path.join('pnn/results',
                                                      suffix_paths)
    path_to_directory_coeffs_vis_kodak = os.path.join('pnn/visualization/checking_predictions/kodak/',
                                                      suffix_paths)
    path_to_directory_coeffs_vis_bsds = os.path.join('pnn/visualization/checking_predictions/bsds/',
                                                     suffix_paths)
    with open(os.path.join('sets/results/training_set/means/', tag_channel, 'mean_training.pkl'), 'rb') as file:
        mean_training = pickle.load(file)
    
    # `tuple_limits` prevents the context for PNN and
    # the reference lines for IPFCN-S from going out of
    # the bounds of the image channel.
    tuple_limits = (max(0, 8 - width_target), 3*width_target)
    row_1sts_kodak = numpy.random.randint(tuple_limits[0],
                                          high=channels_kodak_uint8.shape[1] - tuple_limits[1] + 1,
                                          size=40)
    col_1sts_kodak = numpy.random.randint(tuple_limits[0],
                                          high=channels_kodak_uint8.shape[2] - tuple_limits[1] + 1,
                                          size=40)
    row_1sts_bsds = numpy.random.randint(tuple_limits[0],
                                         high=channels_bsds_uint8.shape[1] - tuple_limits[1] + 1,
                                         size=10)
    col_1sts_bsds = numpy.random.randint(tuple_limits[0],
                                         high=channels_bsds_uint8.shape[2] - tuple_limits[1] + 1,
                                         size=10)
    
    # The entire Tensorflow graph is built in the
    # special method `__init__` of class `PredictionNeuralNetwork`.
    # The portion of the Tensorflow graph dedicated to optimization
    # is not created as it is not needed.
    predictor = pnn.PredictionNeuralNetwork.PredictionNeuralNetwork(batch_size,
                                                                    width_target,
                                                                    is_fully_connected)
    
    # Caffe is imported only if the IPFCN-S is used. If `width_target`
    # is equal to 64, IPFCN-S cannot be used.
    if is_compared_ipfcns and width_target != 64:
        caffe = __import__('caffe')
        if width_target == 32:
            path_to_caffemodel = 'ipfcns/models/ipfcns/IntraFCN205_Size32_iter_1638420.caffemodel'
        else:
            path_to_caffemodel = 'ipfcns/models/ipfcns/IntraFCN205_Size{}_iter_1638700.caffemodel'.format(width_target)
        net_ipfcns_optional = caffe.Net('ipfcns/models/ipfcns/IntraFCN205_deploy_Size{}.prototxt'.format(width_target),
                                        path_to_caffemodel,
                                        caffe.TEST)
    else:
        net_ipfcns_optional = None
    with tf.Session() as sess:
        predict_masks(channels_kodak_uint8,
                      width_target,
                      row_1sts_kodak,
                      col_1sts_kodak,
                      batch_size,
                      mean_training,
                      tuples_width_height_masks_tr,
                      tuples_width_height_masks_val,
                      sess,
                      predictor,
                      net_ipfcns_optional,
                      path_to_directory_coeffs_load_save,
                      path_to_directory_coeffs_vis_kodak)
        predict_masks(channels_bsds_uint8,
                      width_target,
                      row_1sts_bsds,
                      col_1sts_bsds,
                      batch_size,
                      mean_training,
                      tuples_width_height_masks_tr,
                      tuples_width_height_masks_val,
                      sess,
                      predictor,
                      net_ipfcns_optional,
                      path_to_directory_coeffs_load_save,
                      path_to_directory_coeffs_vis_bsds)
    
    # The PNN graph is erased.
    tf.reset_default_graph()

def predict_without_mask_via_ipfcns(channels_uint8, width_target, row_1sts, col_1sts, batch_size, net_ipfcns,
                                    dictionary_performance, coefficient_enlargement, path_to_directory_vis):
    """Predicts each target patch from its reference lines via IPFCN-S.
    
    Parameters
    ----------
    channels_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. `channels_uint8[i, :, :, :]` is the
        channel of the image of index i. `channels_uint8.shape[3]`
        is equal to 1.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row of the 1st pixel of the
        context of index i in the image channel.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column of the 1st pixel of the
        context of index i in the image channel.
    batch_size : int
        Number of predictions IPFCN-S computes in parallel.
    net_ipfcns : Net (class in Caffe)
        IPFCN-S instance.
    dictionary_performance : dict
        Dictionary storing the prediction performance of PNN,
        IPFCN-S and the best HEVC intra prediction mode in terms
        of prediction PSNR. `predict_without_mask_via_ipfcns`
        FILLS THIS DICTIONARY.
    coefficient_enlargement : either int or None
        Coefficient for enlarging an image along
        its first two dimensions before saving it.
    path_to_directory_vis : str
        Path to the directory in which the visualizations
        of the predictions are saved.
    
    """
    # In a given image channel, the difference between
    # the row of the 1st context pixel and the row of the
    # 1st reference lines pixel is equal to `width_target - 8`
    # pixels. The difference between the column of the 1st
    # context pixel and the column of its the reference lines
    # pixel is equal to `width_target - 8` pixels.
    (flattened_pairs_groups_lines_float32, means_float32) = \
            ipfcns.ipfcns.extract_pairs_groups_lines_from_channels_plus_preprocessing(channels_uint8,
                                                                                      width_target,
                                                                                      row_1sts + width_target - 8,
                                                                                      col_1sts + width_target - 8)
    tiled_means_float32 = numpy.tile(numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(means_float32, 1), 2), 3),
                                     (1, width_target, width_target, 1))
    predictions_ipfcns_float32 = ipfcns.ipfcns.predict_by_batch_via_ipfcns(flattened_pairs_groups_lines_float32,
                                                                           net_ipfcns,
                                                                           width_target,
                                                                           batch_size)
    predictions_ipfcns_uint8 = tls.cast_float_to_uint8(predictions_ipfcns_float32 + tiled_means_float32)
    (dictionary_performance['psnrs_ipfcns'], dictionary_performance['frequency_win_ipfcns']) = \
        compute_performance_neural_network_vs_hevc_best_mode(targets_uint8,
                                                             predictions_ipfcns_uint8,
                                                             dictionary_performance['psnrs_hevc_best_mode'])
    tls.histogram(dictionary_performance['psnrs_ipfcns'] - dictionary_performance['psnrs_hevc_best_mode'],
                  'Difference in PSNRs',
                  os.path.join(path_to_directory_vis, 'difference_psnrs_ipfcns_hevc_best_mode.png'))
    for i in range(20):
        ipfcns.ipfcns.visualize_flattened_pair_groups_lines(flattened_pairs_groups_lines_float32[i, :],
                                                            width_target,
                                                            means_float32[i].item(),
                                                            os.path.join(path_to_directory_vis, 'reference_lines_{}.png'.format(i)),
                                                            coefficient_enlargement=coefficient_enlargement)
        tls.save_image(os.path.join(path_to_directory_vis, 'prediction_ipfcns_{}.png'.format(i)),
                       numpy.squeeze(predictions_ipfcns_uint8[i, :, :, :], axis=2),
                       coefficient_enlargement=coefficient_enlargement)

if __name__ == '__main__':
    description = 'Compares the predictions provided by PNN, the predictions provided by IPFCN-S ' \
                  + 'and the predictions provided by the best HEVC intra prediction mode in terms ' \
                  + 'of prediction PSNR.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--width_target',
                        help='width of the target patch',
                        type=parsing.parsing.int_strictly_positive,
                        default=None)
    parser.add_argument('--index_channel',
                        help='channel index',
                        type=parsing.parsing.int_positive,
                        default=None)
    parser.add_argument('--coeff_l2_norm_pred_error',
                        help='coefficient that scales the l2-norm prediction error w.r.t the gradient error and the l2-norm weight decay',
                        type=parsing.parsing.float_positive,
                        default=None)
    parser.add_argument('--coeff_grad_error',
                        help='coefficient that scales the gradient error w.r.t the l2-norm prediction error and the l2-norm weight decay',
                        type=parsing.parsing.float_positive,
                        default=None)
    parser.add_argument('--is_fully_connected',
                        help='is PNN fully-connected?',
                        action='store_true',
                        default=False)
    parser.add_argument('--is_pair',
                        help='if given, the contexts in the training sets for training the different PNNs contain HEVC compression artifacts',
                        action='store_true',
                        default=False)
    parser.add_argument('--is_compared_ipfcns',
                        help='if given, PNN is compared with IPFCN-S',
                        action='store_true',
                        default=False)
    parser.add_argument('--all',
                        help='if given, all other arguments are ignored and a series of comparisons is run to reproduce the results of the paper',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    
    if not args.all:
        is_error = args.width_target is None \
                   or args.index_channel is None \
                   or args.coeff_l2_norm_pred_error is None \
                   or args.coeff_grad_error is None
        if is_error:
            raise ValueError('If `all` is not given, `width_target`, `index_channel`, `coeff_l2_norm_pred_error`, and `coeff_grad_error` have to be specified.')
    
    # PNN computes few predictions in parallel to save RAM.
    batch_size = 10
    if args.all:
        index_channel = 0
        
        # The unused channels are not kept in RAM.
        channels_kodak_uint8 = numpy.load('sets/results/kodak/kodak.npy')[:, :, :, index_channel:index_channel + 1]
        channels_bsds_uint8 = numpy.load('sets/results/bsds/bsds.npy')[:, :, :, index_channel:index_channel + 1]
        
        # When `width_target` is in (4, 8, 16), both fully-connected
        # and convolutional architectures are involved in the experiment.
        tuple_pairs_width_target_is_fc = (
            (4, True),
            (8, True),
            (16, False),
            (32, False),
            (64, False),
            (4, False),
            (8, False),
            (16, True)
        )
        for pair_width_target_is_fc in tuple_pairs_width_target_is_fc:
            predict_masks_kodak_bsds(channels_kodak_uint8,
                                     channels_bsds_uint8,
                                     pair_width_target_is_fc[0],
                                     index_channel,
                                     1.,
                                     0.,
                                     pair_width_target_is_fc[1],
                                     False,
                                     False,
                                     batch_size)
            if pair_width_target_is_fc[1]:
                str_description = 'fully-connected'
            else:
                str_description = 'convolutional'
            print('The comparison for the width of target patch {0} and the \"{1}\" architecture is done.'.format(pair_width_target_is_fc[0], str_description))
    else:
        
        # The unused channels are not kept in RAM.
        channels_kodak_uint8 = numpy.load('sets/results/kodak/kodak.npy')[:, :, :, args.index_channel:args.index_channel + 1]
        channels_bsds_uint8 = numpy.load('sets/results/bsds/bsds.npy')[:, :, :, args.index_channel:args.index_channel + 1]
        predict_masks_kodak_bsds(channels_kodak_uint8,
                                 channels_bsds_uint8,
                                 args.width_target,
                                 args.index_channel,
                                 args.coeff_l2_norm_pred_error,
                                 args.coeff_grad_error,
                                 args.is_fully_connected,
                                 args.is_pair,
                                 args.is_compared_ipfcns,
                                 batch_size)
    print('The results are stored in dictionaries saved in Pickle files in the directory at "pnn/visualization/checking_predictions/".')


