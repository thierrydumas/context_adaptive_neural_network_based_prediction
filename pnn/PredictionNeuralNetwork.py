"""A library defining the class `PredictionNeuralNetwork`."""

import tensorflow as tf

import sets.reading
import pnn.components

NB_ITERS_TRAINING = 800000


class PredictionNeuralNetwork(object):
    """Prediction neural network class.
    
    The attributes are the nodes we need to fetch
    for running the graph of PNN.
    
    """
    
    def __init__(self, batch_size, width_target, is_fully_connected, tuple_coeffs=None, dict_reading=None):
        """Builds the graph of PNN.
        
        Parameters
        ----------
        batch_size : int
            Batch size.
        width_target : int
            Width of the target patch.
        is_fully_connected : bool
            Is PNN fully-connected?
        tuple_coeffs : tuple, optional
            This tuple contains two floats. The 1st float is the
            coefficient that scales the l2-norm prediction error
            with respect to the gradient error and the l2-norm
            weight decay. The 2nd float is the coefficient that
            scales the gradient error with respect to the l2-norm
            prediction error and the l2-norm weight decay. By default,
            `tuple_coeffs` is None, and the graph of PNN does not
            contain the portion dedicated to the optimization of
            the parameters of PNN.
        dict_reading : dict, optional
            By default, `dict_reading` is None, and the graph of PNN
            is dedicated to either validation or test. If `dict_reading`
            contains the 5 keys below, the graph of PNN is dedicated
            to training.
                path_to_directory_threads : str
                    Path to the directory whose subdirectories,
                    named {"thread_i", i = 0 ... `nb_threads` - 1},
                    store the files ".tfrecord" for training.
                nb_threads : int
                    Number of threads.
                mean_training : float
                    Mean pixels intensity computed over the same channel
                    of different YCbCr images.
                tuple_width_height_masks : tuple
                    The 1st integer in this tuple is the width of
                    the mask that covers the right side of the context
                    portion located above the target patch. The 2nd
                    integer in this tuple is the height of the mask
                    that covers the bottom of the context portion
                    located on the left side of the target patch.
                    If `tuple_width_height_masks` is empty, each
                    integer is uniformly drawn.
                is_pair : bool
                    Was there a compression via HEVC when writing
                    the files ".tfrecord"?
        
        Raises
        ------
        ValueError
            If `dict_reading` is not None while `tuple_coeffs` is None.
        
        """
        # It makes no sense to load data from binary files when
        # the purpose is not to train PNN.
        if dict_reading is not None and tuple_coeffs is None:
            raise ValueError('`dict_reading` is not None while `tuple_coeffs` is None')
        self.is_fully_connected = is_fully_connected
        is_small_target = width_target <= 8
        
        # Below is the loading of data.
        if dict_reading is not None:
            
            # When the width of the target patch is smaller than 8,
            # the extraction of target patches, each paired with its
            # two context portions, is offline.
            node_list_batches_float32 = sets.reading.build_batch_training(dict_reading['path_to_directory_threads'],
                                                                          batch_size,
                                                                          dict_reading['nb_threads'],
                                                                          width_target,
                                                                          (not is_small_target, dict_reading['is_pair']),
                                                                          dict_reading['mean_training'],
                                                                          dict_reading['tuple_width_height_masks'],
                                                                          self.is_fully_connected)
            if self.is_fully_connected:
                [self.node_flattened_contexts_float32, self.node_targets_float32] = node_list_batches_float32
            else:
                [self.node_portions_above_float32, self.node_portions_left_float32, self.node_targets_float32] = node_list_batches_float32
        else:
            if self.is_fully_connected:
                self.node_flattened_contexts_float32 = tf.placeholder(tf.float32,
                                                                      shape=(batch_size, 5*width_target**2),
                                                                      name='node_flattened_context')
            else:
                self.node_portions_above_float32 = tf.placeholder(tf.float32,
                                                                  shape=(batch_size, width_target, 3*width_target, 1),
                                                                  name='node_portion_above')
                self.node_portions_left_float32 = tf.placeholder(tf.float32,
                                                                 shape=(batch_size, 2*width_target, width_target, 1),
                                                                 name='node_portion_left')
            
            # If the graph of PNN does not contain the portion
            # dedicated to the optimization of the parameters of
            # PNN, the target patches are not needed.
            if tuple_coeffs is not None:
                self.node_targets_float32 = tf.placeholder(tf.float32,
                                                           shape=(batch_size, width_target, width_target, 1))
        
        # Below is the prediction of each target patch via PNN.
        if self.is_fully_connected:
            scope = 'fully_connected'
            self.node_predictions_float32 = pnn.components.inference_fully_connected(self.node_flattened_contexts_float32,
                                                                                     width_target,
                                                                                     scope)
        else:
            scope = 'convolutional'
            dict_strides_branch = {
                4: (1, 1),
                8: (2, 1),
                16: (2, 1, 2, 1),
                32: (2, 2, 1, 2, 1),
                64: (2, 2, 2, 2, 1)
            }
            self.strides_branch = dict_strides_branch[width_target]
            self.node_predictions_float32 = pnn.components.inference_convolutional(self.node_portions_above_float32,
                                                                                   self.node_portions_left_float32,
                                                                                   self.strides_branch,
                                                                                   scope)
        
        # Below is the optimization of the parameters of PNN.
        if tuple_coeffs is not None:
            (coeff_l2_norm_pred_error, coeff_grad_error) = tuple_coeffs
            
            # When the width of the target patch is smaller than
            # 8, the learning rate is 4 times smaller.
            if is_small_target:
                learning_rates = [1.e-4, 1.e-5, 1.e-6, 1.e-7]
            else:
                learning_rates = [4.e-4, 4.e-5, 4.e-6, 4.e-7]
            with tf.variable_scope('learning_rate'):
                global_step = tf.get_variable('global_step',
                                              dtype=tf.int32,
                                              initializer=0,
                                              trainable=False)
            self.node_learning_rate = tf.train.piecewise_constant(
                global_step,
                [NB_ITERS_TRAINING//2, 3*NB_ITERS_TRAINING//4, 7*NB_ITERS_TRAINING//8],
                learning_rates
            )
            (self.node_dict_errors, self.node_weight_decay, self.node_optimization) = \
                pnn.components.optimizer(self.node_targets_float32,
                                         self.node_predictions_float32,
                                         self.node_learning_rate,
                                         global_step,
                                         coeff_l2_norm_pred_error,
                                         coeff_grad_error,
                                         scope)
        
        # `self.node_saver` is used to save the parameters of PNN.
        self.node_saver = tf.train.Saver(max_to_keep=1000)
    
    def get_global_step(self):
        """Returns the number of updates of the parameters of PNN.
        
        Returns
        -------
        int
            Number of updates of the parameters of PNN
            since the beginning of the 1st training.
        
        """
        with tf.variable_scope('learning_rate', reuse=True):
            return tf.get_variable('global_step', dtype=tf.int32).eval().item()
    
    def initialization(self, sess, path_to_restore):
        """Either initializes all variables or restores a previous model.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        path_to_restore : str
            Path to a previous model. If it is an
            empty string, all variables are initialized.
            The path ends with ".ckpt".
        
        """
        if path_to_restore:
            self.node_saver.restore(sess, path_to_restore)
        else:
            tf.global_variables_initializer().run()


