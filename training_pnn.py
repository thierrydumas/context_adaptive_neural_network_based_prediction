"""A script to train PNN."""

import argparse
import os
import pickle
import tensorflow as tf
import time

import parsing.parsing
import pnn.PredictionNeuralNetwork
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains PNN.')
    parser.add_argument('path_to_directory_training_sets',
                        help='path to the directory storing the different training sets (see the script "creating_training_set.py" for further details)')
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
    parser.add_argument('iter_training_resuming',
                        help='training iteration from which the training resumes',
                        type=parsing.parsing.int_positive)
    parser.add_argument('--is_fully_connected',
                        help='is PNN fully-connected?',
                        action='store_true',
                        default=False)
    parser.add_argument('--is_pair',
                        help='if given, the contexts in the training set contain HEVC compression artifacts',
                        action='store_true',
                        default=False)
    parser.add_argument('--batch_size',
                        help='batch size',
                        type=parsing.parsing.int_strictly_positive,
                        default=100,
                        metavar='')
    parser.add_argument('--nb_threads',
                        help='number of threads for building batches',
                        type=parsing.parsing.int_strictly_positive,
                        default=4,
                        metavar='')
    parser.add_argument('--nb_iters_snapshot',
                        help='number of training iterations between two successive snapshots of the PNN model',
                        type=parsing.parsing.int_strictly_positive,
                        default=10000,
                        metavar='')
    args = parser.parse_args()
    
    if args.width_target <= 8:
        tag_width_target = 'width_target_{}'.format(args.width_target)
    else:
        tag_width_target = 'width_target_None'
    if args.is_fully_connected:
        tag_arch = 'fully_connected'
    else:
        tag_arch = 'convolutional'
    
    # The PNN models resulting from the training using contexts
    # with HEVC compression artifacts are separated from those
    # resulting from the training using contexts without HEVC
    # compression artifact using `tag_pair`.
    if args.is_pair:
        tag_pair = 'pair'
    else:
        tag_pair = 'single'
    
    # The PNN models resulting from the training on the luminance
    # channel of YCbCr images are separated from those resulting
    # from the training on a chrominance channel of YCbCr images
    # using `tag_channel`.
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
    
    # The PNN models resulting from the training with
    # random masks are separated from those resulting
    # from the training with fixed masks using `tag_masks_tr`.
    if args.tuple_width_height_masks_tr:
        tag_masks_tr = 'masks_tr_{0}_{1}'.format(args.tuple_width_height_masks_tr[0],
                                                 args.tuple_width_height_masks_tr[1])
    else:
        tag_masks_tr = 'masks_tr_random'
    path_to_directory_threads = os.path.join(args.path_to_directory_training_sets,
                                             tag_width_target,
                                             tag_pair,
                                             tag_channel)
    path_to_directory_load_save = os.path.join('pnn/results',
                                               'width_target_{}'.format(args.width_target),
                                               tag_arch,
                                               tag_pair,
                                               tag_channel,
                                               tag_coeffs,
                                               tag_masks_tr)
    if not os.path.isdir(path_to_directory_load_save):
        os.makedirs(path_to_directory_load_save)
    
    # If `args.iter_training_resuming` is not equal to 0,
    # the current training portion is not the 1st training
    # portion.
    if args.iter_training_resuming:
        path_to_restore = os.path.join(path_to_directory_load_save,
                                       'model_{}.ckpt'.format(args.iter_training_resuming))
    else:
        path_to_restore = ''
    with open(os.path.join('sets/results/training_set/means/', tag_channel, 'mean_training.pkl'), 'rb') as file:
        mean_training = pickle.load(file)
    dict_reading = {
        'path_to_directory_threads': path_to_directory_threads,
        'nb_threads': args.nb_threads,
        'mean_training': mean_training,
        'tuple_width_height_masks': args.tuple_width_height_masks_tr,
        'is_pair': args.is_pair
    }
    
    # The entire Tensorflow graph is built in the
    # special method `__init__` of class `PredictionNeuralNetwork`.
    predictor = pnn.PredictionNeuralNetwork.PredictionNeuralNetwork(args.batch_size,
                                                                    args.width_target,
                                                                    args.is_fully_connected,
                                                                    tuple_coeffs=(args.coeff_l2_norm_pred_error, args.coeff_grad_error),
                                                                    dict_reading=dict_reading)
    
    # `nb_iters_remaining` is the number of training iterations
    # left before reaching `pnn.PredictionNeuralNetwork.NB_ITERS_TRAINING`
    # training iterations.
    nb_iters_remaining = pnn.PredictionNeuralNetwork.NB_ITERS_TRAINING - args.iter_training_resuming
    t_start = time.time()
    coordinator = tf.train.Coordinator()
    with tf.Session() as sess:
        predictor.initialization(sess,
                                 path_to_restore)
        list_threads = tf.train.start_queue_runners(coord=coordinator)
        for i in range(nb_iters_remaining):
            sess.run(predictor.node_optimization)
            if (i + 1) % args.nb_iters_snapshot == 0 or args.iter_training_resuming + i == pnn.PredictionNeuralNetwork.NB_ITERS_TRAINING - 1:
                global_step = predictor.get_global_step()
                print('\nNumber of training iteration since the beginning of the 1st training portion: {}'.format(global_step))
                print('Learning rate: {}'.format(round(sess.run(predictor.node_learning_rate).item(), 7)))
                predictor.node_saver.save(sess,
                                          os.path.join(path_to_directory_load_save, 'model_{}.ckpt'.format(global_step)))
        
        # The coordinate must stop the threads
        # before the current session is closed.
        coordinator.request_stop()
        coordinator.join(list_threads)
    t_stop = time.time()
    nb_hours = int((t_stop - t_start)/3600)
    nb_minutes = int((t_stop - t_start)/60)
    print('\nThe training portion toke {0} hours and {1} minutes.'.format(nb_hours, nb_minutes - 60*nb_hours))


