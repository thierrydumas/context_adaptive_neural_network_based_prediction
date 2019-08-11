"""A script to check the training of PNN.

The training of PNN is checked using the training subset and the BSDS test set.

"""

import argparse
import numpy
import os
import pickle
import tensorflow as tf

import parsing.parsing
import pnn.PredictionNeuralNetwork
import pnn.visualization
import sets.common
import tools.tools as tls

def visualize_weights_pnn(predictor, path_to_directory_vis):
    """Visualizes the weights in several layers of PNN.
    
    Parameters
    ----------
    predictor : PredictionNeuralNetwork
        PNN instance.
    path_to_directory_vis : str
        Path to the directory in which the visualizations
        of the weights of PNN are saved.
    
    """
    if predictor.is_fully_connected:
        with tf.variable_scope('fully_connected', reuse=True):
            weights_0 = tf.get_variable('weights_0', dtype=tf.float32).eval()
            weights_3 = tf.get_variable('weights_3', dtype=tf.float32).eval()
        pnn.visualization.visualize_weights_fully_connected_1st(weights_0,
                                                                20,
                                                                os.path.join(path_to_directory_vis, 'image_weights_0.png'))
        pnn.visualization.visualize_weights_fully_connected_last(weights_3,
                                                                 20,
                                                                 os.path.join(path_to_directory_vis, 'image_weights_3.png'))
    else:
        
        # The number of transpose convolutional layers
        # in the merger of PNN changes with the width
        # of the target patch.
        index_tconv = len(predictor.strides_branch) - 1
        with tf.variable_scope('convolutional', reuse=True):
            with tf.variable_scope('branch_above', reuse=True):
                with tf.variable_scope('convolution_0', reuse=True):
                    weights_branch_above_0 = tf.get_variable('weights', dtype=tf.float32).eval()
            with tf.variable_scope('branch_left', reuse=True):
                with tf.variable_scope('convolution_0', reuse=True):
                    weights_branch_left_0 = tf.get_variable('weights', dtype=tf.float32).eval()
            with tf.variable_scope('merger', reuse=True):
                with tf.variable_scope('transpose_convolution_{}'.format(index_tconv, reuse=True)):
                    weights_merger_tconv = tf.get_variable('weights', dtype=tf.float32).eval()
        pnn.visualization.visualize_weights_convolutional(weights_branch_above_0,
                                                          8,
                                                          os.path.join(path_to_directory_vis, 'image_weights_branch_above_0.png'))
        pnn.visualization.visualize_weights_convolutional(weights_branch_left_0,
                                                          8,
                                                          os.path.join(path_to_directory_vis, 'image_weights_branch_left_0.png'))
        pnn.visualization.visualize_weights_convolutional(weights_merger_tconv,
                                                          8,
                                                          os.path.join(path_to_directory_vis, 'image_weights_merger_tconv_{}.png'.format(index_tconv)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checks the training of PNN.')
    parser.add_argument('width_target',
                        help='width of the target patch',
                        type=parsing.parsing.int_strictly_positive)
    parser.add_argument('index_channel',
                        help='channel index',
                        type=parsing.parsing.int_positive)
    parser.add_argument('coeff_l2_norm_pred_error',
                        help='coefficient that scales the l2-norm prediction error w.r.t the gradient error and the l2-norm weight decay',
                        type=parsing.parsing.float_positive)
    parser.add_argument('coeff_grad_error',
                        help='coefficient that scales the gradient error w.r.t the l2-norm prediction error and the l2-norm weight decay',
                        type=parsing.parsing.float_positive)
    parser.add_argument('tuple_width_height_masks_tr',
                        help='width of the "above" mask and height of the "left" mask separated by a comma (training phase)',
                        type=parsing.parsing.tuple_two_positive_integers)
    parser.add_argument('tuple_width_height_masks_val',
                        help='width of the "above" mask and height of the "left" mask separated by a comma (validation phase)',
                        type=parsing.parsing.tuple_two_positive_integers)
    parser.add_argument('--is_fully_connected',
                        help='is PNN fully-connected?',
                        action='store_true',
                        default=False)
    parser.add_argument('--is_pair',
                        help='if given, the contexts in the training set for training PNN contain HEVC compression artifacts',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    
    if args.is_fully_connected:
        tag_arch = 'fully_connected'
    else:
        tag_arch = 'convolutional'
    if args.is_pair:
        tag_pair = 'pair'
    else:
        tag_pair = 'single'
    if args.index_channel == 0:
        tag_channel = 'luminance'
    elif args.index_channel == 1:
        tag_channel = 'chrominance_blue'
    elif args.index_channel == 2:
        tag_channel = 'chrominance_red'
    else:
        raise ValueError('`args.index_channel` does not belong to {0, 1, 2}.')
    tag_coeffs = '{0}_{1}'.format(tls.float_to_str(args.coeff_l2_norm_pred_error),
                                  tls.float_to_str(args.coeff_grad_error))
    
    # The width of the mask that covers the right side
    # of the context portion located above the target
    # patch and the height of the masks that covers the
    # bottom of the context portion located on the left
    # side of the target patch are not necessarily the
    # same during the training and validation phases.
    if args.tuple_width_height_masks_tr:
        tag_masks_tr = 'masks_tr_{0}_{1}'.format(args.tuple_width_height_masks_tr[0],
                                                 args.tuple_width_height_masks_tr[1])
    else:
        tag_masks_tr = 'masks_tr_random'
    tag_masks_val = 'masks_val_{0}_{1}'.format(args.tuple_width_height_masks_val[0],
                                               args.tuple_width_height_masks_val[1])
    
    # `suffix_paths` enables to integrate the characteristics
    # of PNN into the path to the directory containing the PNN
    # model and the path to the directory storing visualizations.
    suffix_paths = os.path.join('width_target_{}'.format(args.width_target),
                                tag_arch,
                                tag_pair,
                                tag_channel,
                                tag_coeffs,
                                tag_masks_tr)
    path_to_directory_load_save = os.path.join('pnn/results',
                                               suffix_paths)
    path_to_directory_masks_tr_vis = os.path.join('pnn/visualization/checking_loss_parameters',
                                                  suffix_paths)
    path_to_directory_vis = os.path.join(path_to_directory_masks_tr_vis,
                                         tag_masks_val)
    if not os.path.isdir(path_to_directory_vis):
        os.makedirs(path_to_directory_vis)
    with open(os.path.join('sets/results/training_set/means/', tag_channel, 'mean_training.pkl'), 'rb') as file:
        mean_training = pickle.load(file)
    
    # The contexts that are extracted from the training subset
    # and the BSDS test set have no HEVC compression artifact.
    training_subset_uint8 = numpy.load('sets/results/training_subset/training_subset.npy')[:, :, :, args.index_channel:args.index_channel + 1]
    bsds_uint8 = numpy.load('sets/results/bsds/bsds.npy')[:, :, :, args.index_channel:args.index_channel + 1]
    row_1sts = numpy.array([0, 49],
                           dtype=numpy.int32)
    col_1sts = numpy.array([49, 0],
                           dtype=numpy.int32)
    
    # The entire Tensorflow graph is built in the
    # special method `__init__` of class `PredictionNeuralNetwork`.
    predictor = pnn.PredictionNeuralNetwork.PredictionNeuralNetwork(training_subset_uint8.shape[0]*row_1sts.size,
                                                                    args.width_target,
                                                                    args.is_fully_connected,
                                                                    tuple_coeffs=(args.coeff_l2_norm_pred_error, args.coeff_grad_error))
    tuple_batches_tr_float32 = \
        sets.common.extract_context_portions_targets_from_channels_plus_preprocessing(training_subset_uint8,
                                                                                      args.width_target,
                                                                                      row_1sts,
                                                                                      col_1sts,
                                                                                      mean_training,
                                                                                      args.tuple_width_height_masks_val,
                                                                                      predictor.is_fully_connected)
    tuple_batches_val_float32 = \
        sets.common.extract_context_portions_targets_from_channels_plus_preprocessing(bsds_uint8,
                                                                                      args.width_target,
                                                                                      row_1sts,
                                                                                      col_1sts,
                                                                                      mean_training,
                                                                                      args.tuple_width_height_masks_val,
                                                                                      predictor.is_fully_connected)
    
    # The checking takes into account all PNN
    # models inside the directory at `path_to_directory_load_save`.
    # In Tensorflow 0.x and Tensorflow 1.x, when a
    # model is saved during the training phase, a
    # file of extension ".ckpt.meta" is saved.
    iters_training = tls.collect_integer_between_tags_in_each_filename(path_to_directory_load_save,
                                                                       ('.ckpt.meta',),
                                                                       'model_',
                                                                       '.ckpt')
    if iters_training:
        nb_saved_models = len(iters_training)
        
        # If `args.coeff_l2_norm_pred_error` is equal to 0.0,
        # the computation of the l2-norm prediction error does
        # not exist in the graph of PNN. Similarly, if `args.coeff_grad_error`
        # is equal to 0.0, the computation of the gradient error
        # does not exist in the graph of PNN.
        if args.coeff_l2_norm_pred_error:
            l2_norm_pred_errors_float64 = numpy.zeros((2, nb_saved_models))
        if args.coeff_grad_error:
            grad_errors_float64 = numpy.zeros((2, nb_saved_models))
        w_decays_float64 = numpy.zeros((1, nb_saved_models))
        global_steps_int32 = numpy.zeros(nb_saved_models,
                                         dtype=numpy.int32)
        with tf.Session() as sess:
            for i, iter_training in enumerate(iters_training):
                path_to_restore = os.path.join(path_to_directory_load_save,
                                               'model_{}.ckpt'.format(iter_training))
                
                # The Tensorflow graph is not modified. But, the parameters
                # of PNN are reset.
                predictor.initialization(sess,
                                         path_to_restore)
                if predictor.is_fully_connected:
                    dict_errors_tr = sess.run(
                        predictor.node_dict_errors,
                        feed_dict={
                            predictor.node_flattened_contexts_float32:tuple_batches_tr_float32[0],
                            predictor.node_targets_float32:tuple_batches_tr_float32[1]
                        }
                    )
                    dict_errors_val = sess.run(
                        predictor.node_dict_errors,
                        feed_dict={
                            predictor.node_flattened_contexts_float32:tuple_batches_val_float32[0],
                            predictor.node_targets_float32:tuple_batches_val_float32[1]
                        }
                    )
                else:
                    dict_errors_tr = sess.run(
                        predictor.node_dict_errors,
                        feed_dict={
                            predictor.node_portions_above_float32:tuple_batches_tr_float32[0],
                            predictor.node_portions_left_float32:tuple_batches_tr_float32[1],
                            predictor.node_targets_float32:tuple_batches_tr_float32[2]
                        }
                    )
                    dict_errors_val = sess.run(
                        predictor.node_dict_errors,
                        feed_dict={
                            predictor.node_portions_above_float32:tuple_batches_val_float32[0],
                            predictor.node_portions_left_float32:tuple_batches_val_float32[1],
                            predictor.node_targets_float32:tuple_batches_val_float32[2]
                        }
                    )
                if args.coeff_l2_norm_pred_error:
                    l2_norm_pred_errors_float64[0, i] = dict_errors_tr['l2_norm_pred_error']
                    l2_norm_pred_errors_float64[1, i] = dict_errors_val['l2_norm_pred_error']
                if args.coeff_grad_error:
                    grad_errors_float64[0, i] = dict_errors_tr['grad_error']
                    grad_errors_float64[1, i] = dict_errors_val['grad_error']
                w_decays_float64[0, i] = sess.run(predictor.node_weight_decay)
                global_steps_int32[i] = predictor.get_global_step()
            
            # The weights of the PNN model with the longest training
            # time are displayed.
            visualize_weights_pnn(predictor,
                                  path_to_directory_masks_tr_vis)
        
        if args.coeff_l2_norm_pred_error:
            tls.plot_graphs(global_steps_int32,
                            l2_norm_pred_errors_float64,
                            'iterations',
                            'l2-norm prediction error',
                            'Evolution of the l2-norm prediction error over iterations',
                            os.path.join(path_to_directory_vis, 'l2_norm_prediction_error.png'),
                            legend=['training', 'validation'])
        if args.coeff_grad_error:
            tls.plot_graphs(global_steps_int32,
                            grad_errors_float64,
                            'iterations',
                            'gradient error',
                            'Evolution of the gradient error over iterations',
                            os.path.join(path_to_directory_vis, 'gradient_error.png'),
                            legend=['training', 'validation'])
        tls.plot_graphs(global_steps_int32,
                        w_decays_float64,
                        'iterations',
                        'weight decay',
                        'Evolution of the weight decay over iterations',
                        os.path.join(path_to_directory_masks_tr_vis, 'weight_decay.png'))
    else:
        print('There is no PNN model in the directory at "{}".'.format(path_to_directory_load_save))


