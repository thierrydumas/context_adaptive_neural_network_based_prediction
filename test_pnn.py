"""A script to test all the libraries in the directory "pnn"."""

import argparse
import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

import pnn.batching
import pnn.components
import pnn.PredictionNeuralNetwork
import pnn.tfutils
import pnn.visualization
import tools.tools as tls


class TesterPNN(object):
    """Class for testing all the libraries in the directory "pnn"."""
    
    def test_branch(self):
        """Tests the function `branch` in the file "pnn/components.py".
        
        The test is successful if the shape of the output
        of the branch is equal to [5, 8, 16, 128].
        
        """
        strides_branch = (2, 2, 1)
        
        node_input = tf.placeholder(tf.float32,
                                    shape=(5, 32, 64, 1))
        node_output = pnn.components.branch(node_input,
                                            strides_branch,
                                            'branch')
        print('Shape of the output of the branch:')
        print(node_output.get_shape().as_list())
    
    def test_channelwise_fully_connected_merger(self):
        """Tests the function `channelwise_fully_connected_merger` in the file "pnn/tfutils.py".
        
        The test is successful if only the feature map
        of index 0 for the example of index 0 in the output
        of the channelwise fully-connected merger is not
        filled with zeros.
        
        """
        batch_size = 2
        height_input_0 = 2
        width_input_0 = 3
        height_input_1 = 4
        width_input_1 = 5
        nb_maps = 3
        height_output = 3
        width_output = 2
        
        input_0_float32 = numpy.zeros((batch_size, height_input_0, width_input_0, nb_maps),
                                      dtype=numpy.float32)
        input_0_float32[0, :, :, 0] = 1.
        input_1_float32 = numpy.zeros((batch_size, height_input_1, width_input_1, nb_maps),
                                      dtype=numpy.float32)
        input_1_float32[0, :, :, 0] = 1.
        node_input_0 = tf.placeholder(tf.float32,
                                      shape=(batch_size, height_input_0, width_input_0, nb_maps))
        node_input_1 = tf.placeholder(tf.float32,
                                      shape=(batch_size, height_input_1, width_input_1, nb_maps))
        node_output = pnn.tfutils.channelwise_fully_connected_merger(node_input_0,
                                                                     node_input_1,
                                                                     height_output,
                                                                     width_output,
                                                                     'channelwise_fully_connected_merger')
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            output_float32 = sess.run(
                node_output,
                feed_dict={
                    node_input_0:input_0_float32,
                    node_input_1:input_1_float32
                }
            )
        for i in range(batch_size):
            for j in range(nb_maps):
                print('Feature map of index {0} for the example of index {1} in the output of the channelwise fully-connected merger:'.format(j ,i))
                print(output_float32[i, :, :, j])
    
    def test_convolution_2d_auto_width_kernels(self):
        """Tests the function `convolution_2d_auto_width_kernels` in the file "pnn/tfutils.py".
        
        The test is successful if the height and the width
        of the input to the convolutional layer are equal to
        respectively the height and the width of the output of
        the convolutional layer multiplied by the convolutional
        stride.
        
        """
        nb_maps_output = 12
        stride = 3
        
        node_input = tf.placeholder(tf.float32,
                                    shape=(2, 9, 6, 5))
        node_output = pnn.tfutils.convolution_2d_auto_width_kernels(node_input,
                                                                    nb_maps_output,
                                                                    stride,
                                                                    False,
                                                                    'convolution_2d')
        print('Convolutional stride: {}'.format(stride))
        print('Shape of the input to the convolutional layer:')
        print(node_input.get_shape().as_list())
        print('Shape of the output of the convolutional layer:')
        print(node_output.get_shape().as_list())
    
    def test_difference_gradients_direction(self):
        """Tests the function `difference_gradients_direction` in the file "pnn/tfutils.py".
        
        The test is successful if the difference of gradients is correct.
        
        """
        array_0_term_minus = numpy.random.randint(-5,
                                                  high=5,
                                                  size=(3, 4)).astype(numpy.float32)
        array_0_term_plus = numpy.random.randint(-5,
                                                 high=5,
                                                 size=(3, 4)).astype(numpy.float32)
        array_1_term_minus = numpy.random.randint(-5,
                                                  high=5,
                                                  size=(3, 4)).astype(numpy.float32)
        array_1_term_plus = numpy.random.randint(-5,
                                                 high=5,
                                                 size=(3, 4)).astype(numpy.float32)
        node_tensor_0_term_minus = tf.placeholder(tf.float32,    
                                                  shape=(3, 4))
        node_tensor_0_term_plus = tf.placeholder(tf.float32,
                                                 shape=(3, 4))
        node_tensor_1_term_minus = tf.placeholder(tf.float32,
                                                  shape=(3, 4))
        node_tensor_1_term_plus = tf.placeholder(tf.float32,
                                                 shape=(3, 4))
        node_diffs = pnn.tfutils.difference_gradients_direction(node_tensor_0_term_minus,
                                                                node_tensor_0_term_plus,
                                                                node_tensor_1_term_minus,
                                                                node_tensor_1_term_plus)
        with tf.Session() as sess:
            diffs = sess.run(
                node_diffs,
                feed_dict={
                    node_tensor_0_term_minus:array_0_term_minus,
                    node_tensor_0_term_plus:array_0_term_plus,
                    node_tensor_1_term_minus:array_1_term_minus,
                    node_tensor_1_term_plus:array_1_term_plus
                }
            )
        print('"minus" term for the 1st array:')
        print(array_0_term_minus)
        print('"plus" term for the 1st array:')
        print(array_0_term_plus)
        print('"minus" term for the 2nd array:')
        print(array_1_term_minus)
        print('"plus" term for the 2nd array:')
        print(array_1_term_plus)
        print('Difference of gradients:')
        print(diffs)
    
    def test_inference_convolutional(self):
        """Tests the function `inference_convolutional` in the file "pnn/components.py".
        
        The test is successful if the list of trainable
        variables name is ["convolutional/branch_above/convolution_0/weights:0",
        "convolutional/branch_above/convolution_0/biases:0",
        "convolutional/branch_above/convolution_1/weights:0",
        "convolutional/branch_above/convolution_1/biases:0",
        "convolutional/branch_left/convolution_0/weights:0",
        "convolutional/branch_left/convolution_0/biases:0",
        "convolutional/branch_left/convolution_1/weights:0",
        "convolutional/branch_left/convolution_1/biases:0",
        "convolutional/merger/channelwise_fully_connected_merger/weights:0",
        "convolutional/merger/channelwise_fully_connected_merger/biases:0",
        "convolutional/merger/transpose_convolutional_0/weights:0",
        "convolutional/merger/transpose_convolutional_0/biases:0",
        "convolutional/merger/transpose_convolutional_1/weights:0",
        "convolutional/merger/transpose_convolutional_1/biases:0"].
        Besides, the shape of the prediction of the
        target patches is equal to [5, 16, 16, 1].
        
        """
        width_target = 16
        strides_branch = (2, 2)
        
        node_portions_above_float32 = tf.placeholder(tf.float32,
                                                     shape=(5, width_target, 3*width_target, 1))
        node_portions_left_float32 = tf.placeholder(tf.float32,
                                                    shape=(5, 2*width_target, width_target, 1))
        node_predictions_float32 = pnn.components.inference_convolutional(node_portions_above_float32,
                                                                          node_portions_left_float32,
                                                                          strides_branch,
                                                                          'convolutional')
        list_names = []
        for trainable_variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            list_names.append(trainable_variable.name)
        print('List of trainable variables name:')
        print(list_names)
        print('Shape of the prediction of the target patches:')
        print(node_predictions_float32.get_shape().as_list())
    
    def test_inference_fully_connected(self):
        """Tests the function `inference_fully_connected` in the file "pnn/components.py".
        
        The test is successful if the list of trainable
        variables name is ["fully_connected/weights:0",
        "fully_connected/biases:0", "fully_connected/weights:1",
        "fully_connected/biases:1", "fully_connected/weights:2",
        "fully_connected/biases:2", "fully_connected/weights:3",
        "fully_connected/biases:3"]. Besides, the shape
        of the prediction of the target patches is [5, 4, 4, 1].
        
        """
        width_target = 4
        
        node_flattened_contexts_float32 = tf.placeholder(tf.float32,
                                                         shape=(5, 5*width_target**2))
        node_predictions_float32 = pnn.components.inference_fully_connected(node_flattened_contexts_float32,
                                                                            width_target,
                                                                            'fully_connected')
        list_names = []
        for trainable_variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            list_names.append(trainable_variable.name)
        print('List of trainable variables name:')
        print(list_names)
        print('Shape of the prediction of the target patches:')
        print(node_predictions_float32.get_shape().as_list())
    
    def test_leaky_relu(self):
        """Tests the function `leaky_relu` in the file "pnn/tfutils.py".
        
        A plot is saved at
        "pnn/pseudo_visualization/leaky_relu.png".
        The test is successful if the plot looks
        like that of LeakyReLU with slope 0.1.
        
        """
        input = numpy.linspace(-10.,
                               10.,
                               num=201,
                               dtype=numpy.float32)
        node_input = tf.placeholder(tf.float32,
                                    shape=201)
        node_output = pnn.tfutils.leaky_relu(node_input)
        with tf.Session() as sess:
            output = sess.run(node_output, feed_dict={node_input:input})
        plt.plot(input, output)
        plt.title('Leaky ReLU with slope 0.1')
        plt.savefig('pnn/pseudo_visualization/leaky_relu.png')
        plt.clf()
    
    def test_merger(self):
        """Tests the function `merger` in the file "pnn/components.py".
        
        The test is successful if the list of trainable
        variables name is ["merger/channelwise_fully_connected_merger/weights:0",
        "merger/channelwise_fully_connected_merger/biases:0",
        "merger/transpose_convolutional_0/weights:0",
        "merger/transpose_convolutional_0/biases:0",
        "merger/transpose_convolutional_1/weights:0",
        "merger/transpose_convolutional_1/biases:0"].
        Moreover, the shape of the output of the merger
        is equal to [5, 8, 8, 1].
        
        """
        width_target = 8
        strides_merger = (2, 2)
        
        node_input_0 = tf.placeholder(tf.float32,
                                      shape=(5, 4, 8, 64))
        node_input_1 = tf.placeholder(tf.float32,
                                      shape=(5, 12, 24, 64))
        node_output = pnn.components.merger(node_input_0,
                                            node_input_1,
                                            width_target,
                                            strides_merger,
                                            'merger')
        list_names = []
        for trainable_variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            list_names.append(trainable_variable.name)
        print('List of trainable variables name:')
        print(list_names)
        print('Shape of the output of the merger:')
        print(node_output.get_shape().as_list())
    
    def test_optimizer(self):
        """Tests the function `optimizer` in the file "pnn/components.py".
        
        The test is successful if the dictionary of errors
        contains a single key: "l2_norm_pred_error".
        
        """
        learning_rate = 1.e-4
        scope = 'fully_connected'
        
        global_step = tf.get_variable('global_step',
                                      dtype=tf.int32,
                                      initializer=0,
                                      trainable=False)
        with tf.variable_scope(scope):
            weights = tf.get_variable('weights',
                                      dtype=tf.float32,
                                      initializer=tf.random_normal([80, 100],
                                                                   mean=0.,
                                                                   stddev=0.01,
                                                                   dtype=tf.float32))
        node_targets_float32 = tf.placeholder(tf.float32,
                                              shape=(5, 8, 8, 1))
        node_predictions_float32 = tf.placeholder(tf.float32,
                                                  shape=(5, 8, 8, 1))
        node_dict_errors = pnn.components.optimizer(node_targets_float32,
                                                    node_predictions_float32,
                                                    learning_rate,
                                                    global_step,
                                                    1.,
                                                    0.,
                                                    scope)[0]
        print('Keys in the dictionary of errors:')
        for key in node_dict_errors:
            print(key)
    
    def test_prediction_error_gradient(self):
        """Tests the function `prediction_error_gradient` in the file "pnn/tfutils.py".
        
        The test is successful if the gradient
        error computed by `prediction_error_gradient`
        is equal to the gradient error computed
        by hand.
        
        """
        targets_float32 = numpy.ones((2, 3, 4, 1),
                                     dtype=numpy.float32)
        predictions_float32 = numpy.ones((2, 3, 4, 1),
                                         dtype=numpy.float32)
        predictions_float32[0, 1, 1] = 3.
        node_targets_float32 = tf.placeholder(tf.float32,
                                              shape=(2, 3, 4, 1))
        node_predictions_float32 = tf.placeholder(tf.float32,
                                                  shape=(2, 3, 4, 1))
        node_grad_error = pnn.tfutils.prediction_error_gradient(node_targets_float32,
                                                                node_predictions_float32)
        with tf.Session() as sess:
            grad_error = sess.run(
                node_grad_error,
                feed_dict={
                    node_targets_float32:targets_float32,
                    node_predictions_float32:predictions_float32
                }
            )
        print('Gradient error computed by `prediction_error_gradient`: {}'.format(grad_error))
        print('Gradient error computed by hand: {}'.format(8.))
    
    def test_prediction_error_l2_norm(self):
        """Tests the function `prediction_error_l2_norm` in the file "pnn/tfutils.py".
        
        The test is successful if the l2-norm prediction
        error computed by `prediction_error_l2_norm` is
        equal to the l2-norm prediction error computed
        by hand.
        
        """
        targets_float32 = numpy.ones((2, 5, 5, 1),
                                     dtype=numpy.float32)
        targets_float32[0, :, :, :] = 4.*targets_float32[0, :, :, :]
        predictions_float32 = numpy.ones(targets_float32.shape,
                                         dtype=numpy.float32)
        node_targets_float32 = tf.placeholder(tf.float32,
                                              shape=targets_float32.shape)
        node_predictions_float32 = tf.placeholder(tf.float32,
                                                  shape=predictions_float32.shape)
        node_l2_norm_pred_error = pnn.tfutils.prediction_error_l2_norm(node_targets_float32,
                                                                       node_predictions_float32)
        with tf.Session() as sess:
            l2_norm_pred_error = sess.run(
                node_l2_norm_pred_error,
                feed_dict={
                    node_targets_float32:targets_float32,
                    node_predictions_float32:predictions_float32
                }
            )
        print('L2-norm prediction error computed by `prediction_error_l2_norm`: {}'.format(l2_norm_pred_error))
        print('L2-norm prediction error computed by hand: {}'.format(7.5))
    
    def test_prediction_neural_network(self):
        """Tests the class `PredictionNeuralNetwork` in the file "pnn/PredictionNeuralNetwork.py".
        
        The test is successful if the l2-norm prediction error
        after the training is much smaller than the l2-norm
        prediction error before the training.
        
        """
        batch_size = 100
        width_target = 8
        is_fully_connected = True
        tuple_coeffs = (1., 0.)
        
        # `nb_iters_training` is the number of training iterations.
        nb_iters_training = 50000
        
        # The training contexts have no HEVC compression artifact.
        path_to_directory_threads = os.path.join('pnn/pseudo_data/prediction_neural_network/',
                                                 'width_target_{}'.format(width_target),
                                                 'single')
        
        # `mean_training` is a fake mean pixels intensity.
        dict_reading = {
            'path_to_directory_threads': path_to_directory_threads,
            'nb_threads': 2,
            'mean_training': 114.5,
            'tuple_width_height_masks': (),
            'is_pair': False
        }
        
        # The entire Tensorflow graph is built in the
        # special method `__init__` of class `PredictionNeuralNetwork`.
        predictor = pnn.PredictionNeuralNetwork.PredictionNeuralNetwork(batch_size,
                                                                        width_target,
                                                                        is_fully_connected,
                                                                        tuple_coeffs=tuple_coeffs,
                                                                        dict_reading=dict_reading)
        coordinator = tf.train.Coordinator()
        with tf.Session() as sess:
            predictor.initialization(sess, '')
            list_threads = tf.train.start_queue_runners(coord=coordinator)
            
            # For fully-connected PNN, `dict_errors_0` and
            # `dict_errors_1` are evaluated on two different pairs
            # {batch of flattened masked contexts, batch of target
            # patches}. Nevertheless, the comparison between `dict_errors_0`
            # and `dict_errors_1` gives a rough idea regarding the evolution
            # of the training.
            dict_errors_0 = sess.run(predictor.node_dict_errors)
            weight_decay_0 = sess.run(predictor.node_weight_decay)
            print('L2-norm prediction error before the training: {}'.format(dict_errors_0['l2_norm_pred_error']))
            print('L2-norm weight decay before the training: {}'.format(weight_decay_0))
            for _ in range(nb_iters_training):
                sess.run(predictor.node_optimization)
            dict_errors_1 = sess.run(predictor.node_dict_errors)
            weight_decay_1 = sess.run(predictor.node_weight_decay)
            print('L2-norm prediction error after {0} training iterations: {1}'.format(nb_iters_training, dict_errors_1['l2_norm_pred_error']))
            print('L2-norm weight decay after {0} training iterations: {1}'.format(nb_iters_training, weight_decay_1))
            coordinator.request_stop()
            coordinator.join(list_threads)
    
    def test_predict_by_batch_via_pnn(self):
        """Tests the function `predict_by_batch_via_pnn` in the file "pnn/batching.py".
        
        For i = 0 ... 5, an image is saved at
        "pnn/pseudo_visualization/predict_by_batch_via_pnn/prediction_i.png".
        The test is successful if, for i = 0, the
        image contains a bright gray square. For
        i = 1 ... 5, each image contains the same
        gray square.
        
        """
        batch_size = 2
        
        # The PNN model in the file at
        # "pnn/pseudo_data/predict_by_batch_via_pnn/width_target_16/model.ckpt"
        # was trained with the width of target
        # patch equal to 16, random masking and
        # no gradient term in the objective function
        # to be optimized.
        width_target = 16
        is_fully_connected = False
        
        # `mean_training` was equal to 117.89522342
        # when training the above-mentioned PNN model.
        mean_training = 117.89522342
        
        # The entire Tensorflow graph is built in the
        # special method `__init__` of class `PredictionNeuralNetwork`.
        predictor = pnn.PredictionNeuralNetwork.PredictionNeuralNetwork(batch_size,
                                                                        width_target,
                                                                        is_fully_connected)
        if predictor.is_fully_connected:
            flattened_contexts_float32 = numpy.zeros((3*batch_size, 5*width_target**2), dtype=numpy.float32)
            flattened_contexts_float32[0, :] = mean_training
            tuple_batches_float32 = (flattened_contexts_float32,)
        else:
            portions_above_float32 = numpy.zeros((3*batch_size, width_target, 3*width_target, 1), dtype=numpy.float32)
            portions_above_float32[0, :, :, :] = mean_training
            portions_left_float32 = numpy.zeros((3*batch_size, 2*width_target, width_target, 1), dtype=numpy.float32)
            portions_left_float32[0, :, :, :] = mean_training
            tuple_batches_float32 = (portions_above_float32, portions_left_float32)
        with tf.Session() as sess:
            predictor.initialization(sess,
                                     'pnn/pseudo_data/predict_by_batch_via_pnn/width_target_{}/model.ckpt'.format(width_target))
            predictions_float32 = pnn.batching.predict_by_batch_via_pnn(tuple_batches_float32,
                                                                        sess,
                                                                        predictor,
                                                                        batch_size)
        predictions_uint8 = tls.cast_float_to_uint8(predictions_float32 + mean_training)
        path_to_directory_vis = os.path.join('pnn/pseudo_visualization/predict_by_batch_via_pnn/',
                                             'width_target_{}'.format(width_target))
        if not os.path.isdir(path_to_directory_vis):
            os.makedirs(path_to_directory_vis)
        for i in range(3*batch_size):
            tls.save_image(os.path.join(path_to_directory_vis, 'prediction_{}.png'.format(i)),
                           numpy.squeeze(predictions_uint8[i, :, :, :], axis=2))
    
    def test_reshape_vectors_to_channels(self):
        """Tests the function `reshape_vectors_to_channels` in the file "pnn/tfutils.py".
        
        The test is succesful if each vector is
        reshaped to an image channel. There is
        no data mixing between the different vectors
        during the change of shape.
        
        """
        vectors_float32 = numpy.array([[-1., -2., -3., -4., -5., -6.], [1., 2., 3., 4., 5., 6.]],
                                      dtype=numpy.float32)
        node_vectors_float32 = tf.placeholder(tf.float32,
                                              shape=vectors_float32.shape)
        node_channels = pnn.tfutils.reshape_vectors_to_channels(node_vectors_float32,
                                                                3,
                                                                2)
        with tf.Session() as sess:
            channels_float32 = sess.run(
                node_channels,
                feed_dict={node_vectors_float32:vectors_float32}
            )
        print('1st vector:')
        print(vectors_float32[0, :])
        print('Channel of the 1st image:')
        print(channels_float32[0, :, :, 0])
        print('2nd vector:')
        print(vectors_float32[1, :])
        print('Channel of the 2nd image:')
        print(channels_float32[1, :, :, 0])
    
    def test_slice_4_directions(self):
        """Tests the function `slice_4_directions` in the file "pnn/tfutils.py".
        
        The test is successful if the content of
        each slice is consistent with the direction
        of slicing.
        
        """
        array_2d = numpy.array([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
                               dtype=numpy.float32)
        array_2d_expansion = numpy.expand_dims(numpy.expand_dims(array_2d, 0),
                                               axis=3)
        array_4d = numpy.concatenate((array_2d_expansion, 100.*array_2d_expansion),
                                     axis=0)
        node_tensor_4d = tf.placeholder(tf.float32,
                                        shape=array_4d.shape)
        node_tuple_slices = pnn.tfutils.slice_4_directions(node_tensor_4d)
        with tf.Session() as sess:
            tuple_slices = sess.run(
                node_tuple_slices,
                feed_dict={node_tensor_4d:array_4d}
            )
        print('1st example in the reference array:')
        print(array_4d[0, :, :, 0])
        print('2nd example in the reference array:')
        print(array_4d[1, :, :, 0])
        print('Right slice of the 1st example:')
        print(tuple_slices[1][0, :, :, 0])
        print('Right slice of the 2nd example:')
        print(tuple_slices[1][1, :, :, 0])
        print('Bottom slice of the 1st example:')
        print(tuple_slices[4][0, :, :, 0])
        print('Bottom slice of the 2nd example:')
        print(tuple_slices[4][1, :, :, 0])
        print('Top left slice of the 1st example:')
        print(tuple_slices[7][0, :, :, 0])
        print('Top left slice of the 2nd example:')
        print(tuple_slices[7][1, :, :, 0])
    
    def test_transpose_convolution_2d_auto_width_kernels(self):
        """Tests the function `transpose_convolution_2d_auto_width_kernels` in the file "pnn/tfutils.py".
        
        The test is successful if the height and the width
        of the input to the transpose convolutional layer are
        equal to respectively the height and the width of the
        output of the transpose convolutional layer divided
        by the convolutional stride.
        
        """
        nb_maps_output = 12
        stride = 3
        
        node_input = tf.placeholder(tf.float32,
                                    shape=(2, 3, 2, 5))
        node_output = pnn.tfutils.transpose_convolution_2d_auto_width_kernels(node_input,
                                                                              nb_maps_output,
                                                                              stride,
                                                                              False,
                                                                              'transpose_convolution_2d')
        print('Convolutional stride: {}'.format(stride))
        print('Shape of the input to the transpose convolutional layer:')
        print(node_input.get_shape().as_list())
        print('Shape of the output of the transpose convolutional layer:')
        print(node_output.get_shape().as_list())
    
    def test_visualize_context_portions(self):
        """Tests the function `visualize_context_portions` in the file "pnn/visualization.py".
        
        An image is saved at
        "pnn/pseudo_visualization/visualize_context_portions.png".
        The test is successful if, in this image, the
        context portion above the target patch is gray
        and the context portion on the left side of the
        target patch is black. The background is white.
        
        """
        width_target = 32
        
        # `mean_training` is a fake mean pixels intensity.
        mean_training = 114.5
        
        portion_above_float32 = numpy.zeros((width_target, 3*width_target, 1),
                                            dtype=numpy.float32)
        portion_left_float32 = -mean_training*numpy.ones((2*width_target, width_target, 1),
                                                         dtype=numpy.float32)
        pnn.visualization.visualize_context_portions(portion_above_float32,
                                                     portion_left_float32,
                                                     mean_training,
                                                     'pnn/pseudo_visualization/visualize_context_portions.png')
    
    def test_visualize_flattened_context(self):
        """Tests the function `visualize_flattened_context` in the file "pnn/visualization.py".
        
        An image is saved at
        "pnn/pseudo_visualization/visualize_flattened_context.png".
        The test is successful if this image
        shows a gray context whose 2nd row
        is black. The background is white.
        
        """
        width_target = 8
        
        # `mean_training` is a fake mean pixels intensity.
        mean_training = 114.5
        
        flattened_context_float32 = numpy.zeros(5*width_target**2,
                                                dtype=numpy.float32)
        flattened_context_float32[3*width_target:6*width_target] = -mean_training
        pnn.visualization.visualize_flattened_context(flattened_context_float32,
                                                      width_target,
                                                      mean_training,
                                                      'pnn/pseudo_visualization/visualize_flattened_context.png')
    
    def test_visualize_weights_convolutional(self):
        """Tests the function `visualize_weights_convolutional` in the file "pnn/visualization.py".
        
        An image is saved at
        "pnn/pseudo_visualization/visualize_weights_convolutional.png".
        The test is successful if each weight filter
        in this image is random noise, except the 1st
        one which is gray and the last one which is black.
        
        """
        weights_float32 = numpy.random.uniform(low=-1.,
                                               high=1.,
                                               size=(32, 32, 1, 64)).astype(numpy.float32)
        weights_float32[:, :, :, 0] = 0.
        weights_float32[:, :, :, 63] = -1.
        pnn.visualization.visualize_weights_convolutional(weights_float32,
                                                          8,
                                                          'pnn/pseudo_visualization/visualize_weights_convolutional.png')
    
    def test_visualize_weights_fully_connected_last(self):
        """Tests the function `visualize_weights_fully_connected_last` in the file "pnn/visualization.py".
        
        An image is saved at
        "pnn/pseudo_visualization/visualize_weights_fully_connected_last.png".
        The test is successful if, in this image, the
        top-left square and the bottom-left square are
        gray. The top-right square is black. The bottom-right
        square is white.
        
        """
        weights_float32 = numpy.zeros((4, 64),
                                      dtype=numpy.float32)
        weights_float32[0, :] = -3.1
        weights_float32[1, :] = -20.9
        weights_float32[2, :] = 0.
        weights_float32[3, :] = 10.76
        pnn.visualization.visualize_weights_fully_connected_last(weights_float32,
                                                                 2,
                                                                 'pnn/pseudo_visualization/visualize_weights_fully_connected_last.png')
    
    def test_visualize_weights_fully_connected_1st(self):
        """Tests the function `visualize_weights_fully_connected_1st` in the file "pnn/visualization.py".
        
        An image is saved at
        "pnn/pseudo_visualization/visualize_weights_fully_connected_1st.png".
        The test is successful if, in this image,
        the top-right sub-image is the darkest and
        the bottom-right sub-image is the brightest.
        
        """
        weights_float32 = numpy.zeros((320, 4),
                                      dtype=numpy.float32)
        weights_float32[:, 0] = -3.1
        weights_float32[:, 1] = -20.9
        weights_float32[:, 2] = 0.
        weights_float32[:, 3] = 10.76
        pnn.visualization.visualize_weights_fully_connected_1st(weights_float32,
                                                                2,
                                                                'pnn/pseudo_visualization/visualize_weights_fully_connected_1st.png')
    
    def test_weight_l2_norm(self):
        """Tests the function `weight_l2_norm` in the file "pnn/components.py".
        
        The test is successful if the weight l2-norm
        computed by `weight_l2_norm` is equal to the
        weight l2-norm computed by hand.
        
        """
        with tf.variable_scope('fully_connected'):
            weights_0 = tf.get_variable('weights_0',
                                        dtype=tf.float32,
                                        initializer=tf.ones([2, 3, 1, 4], dtype=tf.float32))
            biases_0 = tf.get_variable('biases_0',
                                       dtype=tf.float32,
                                       initializer=5.*tf.ones([20], dtype=tf.float32))
        with tf.variable_scope('fake_scope'):
            weights_1 = tf.get_variable('weights_1',
                                        dtype=tf.float32,
                                        initializer=2.*tf.ones([3, 4, 2, 1], dtype=tf.float32))
        node_weight_l2_norm = pnn.components.weight_l2_norm()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            weight_l2_norm = sess.run(node_weight_l2_norm)
            print('Cumulated weight l2-norm computed by `weight_l2_norm`: {}'.format(weight_l2_norm))
            print('Cumulated weight l2-norm computed by hand: {}'.format(60.))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all the libraries in the directory "pnn".')
    parser.add_argument('name', help='name of the function/method to be tested')
    args = parser.parse_args()
    
    tester = TesterPNN()
    getattr(tester, 'test_' + args.name)()


