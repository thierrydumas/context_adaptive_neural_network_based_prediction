"""A script to freeze the graph of PNN."""

import argparse
import os
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

import parsing.parsing
import tools.tools as tls
import pnn.PredictionNeuralNetwork

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freezes the graph of PNN.')
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
    if args.tuple_width_height_masks_tr:
        tag_masks_tr = 'masks_tr_{0}_{1}'.format(args.tuple_width_height_masks_tr[0],
                                                 args.tuple_width_height_masks_tr[1])
    else:
        tag_masks_tr = 'masks_tr_random'
    
    # `suffix_paths` enables to integrate the characteristics
    # of PNN into the path to the directory containing the PNN
    # model and the path to the directory storing its frozen graph.
    suffix_paths = os.path.join('width_target_{}'.format(args.width_target),
                                tag_arch,
                                tag_pair,
                                tag_channel,
                                tag_coeffs,
                                tag_masks_tr)
    path_to_directory_load_save = os.path.join('pnn/results',
                                               suffix_paths)
    
    # The directory at `path_to_directory_graphs_frozen` stores
    # the frozen graphs used inside the two modified HEVCs.
    path_to_directory_graphs_frozen = os.path.join('pnn/graphs_frozen',
                                                   suffix_paths)
    if not os.path.isdir(path_to_directory_graphs_frozen):
        os.makedirs(path_to_directory_graphs_frozen)
    
    # The integers in the list `iters_training` are
    # sorted in ascending order.
    iters_training = tls.collect_integer_between_tags_in_each_filename(path_to_directory_load_save,
                                                                       ('.ckpt.meta',),
                                                                       'model_',
                                                                       '.ckpt')
    if iters_training:
        
        # It is crucial to point out that the freezed graph
        # is different from the graph during the training phase.
        # Indeed, the freezed graph does not contain the portion
        # of the graph dedicated to the optimization of the parameters
        # of PNN.
        predictor = pnn.PredictionNeuralNetwork.PredictionNeuralNetwork(1,
                                                                        args.width_target,
                                                                        args.is_fully_connected)
        if predictor.is_fully_connected:
            name_node_output = 'fully_connected/node_output'
        else:
            
            # `os.path.join` is not used as the operators scopes
            # in the files ".pbtxt" should not contain backslashs
            # (Windows).
            name_node_output = 'convolutional/merger/' \
                               + 'transpose_convolution_{}/'.format(len(predictor.strides_branch) - 1) \
                               + 'node_output'
        
        # The PNN model with the longest training time is restored.
        path_to_restore = os.path.join(path_to_directory_load_save,
                                       'model_{}.ckpt'.format(iters_training[-1]))
        name_graph_input = 'graph_input.pbtxt'
        path_to_graph_input = os.path.join(path_to_directory_graphs_frozen,
                                           name_graph_input)
        path_to_graph_output = os.path.join(path_to_directory_graphs_frozen,
                                            'graph_output.pbtxt')
        
        # If a file exists at `path_to_graph_output`, it may contain
        # an outdated freezed graph. That is why the outdated freezed
        # graph is suppressed.
        if os.path.isfile(path_to_graph_input):
            os.remove(path_to_graph_input)
        if os.path.isfile(path_to_graph_output):
            os.remove(path_to_graph_output)
        with tf.Session() as sess:
            tf.train.write_graph(sess.graph.as_graph_def(),
                                 path_to_directory_graphs_frozen,
                                 name_graph_input)
        freeze_graph(path_to_graph_input,
                     '',
                     False,
                     path_to_restore,
                     name_node_output,
                     'save/restore_all',
                     'save/Const:0',
                     path_to_graph_output,
                     False,
                     '')
    else:
        print('There is no PNN model in the directory at "{}".'.format(path_to_directory_load_save))


