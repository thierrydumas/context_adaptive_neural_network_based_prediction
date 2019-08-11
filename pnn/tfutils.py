"""A library containing Tensorflow utilities for the graph of PNN."""

import numpy
import tensorflow as tf

# The functions are sorted in alphabetic order.

def channelwise_fully_connected_merger(input_0, input_1, height_output, width_output, scope):
    """Channelwise fully-connects the two input stack of feature maps to the output stack of feature maps.
    
    Parameters
    ----------
    input_0 : Tensor
        4D tensor with data-type `tf.float32`.
        1st input stack of feature maps.
    input_1 : Tensor
        4D tensor with data-type `tf.float32`.
        2nd input stack of feature maps. `input_1.get_shape().as_list()[0]`
        is equal to `input_0.get_shape().as_list()[0]`.
        `input_1.get_shape().as_list()[3]` is equal to
        `input_0.get_shape().as_list()[3]`.
    height_output : int
        Height of the output stack of feature maps.
    width_output : int
        Width of the output stack of feature maps.
    scope : str
        Scope of the weights and biases of the channelwise
        fully-connected merger.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output stack of feature maps. The tensor
        shape is equal to [`input_0.get_shape().as_list()[0]`,
        `height_output`, `width_output`, `input_0.get_shape().as_list()[3]`].
    
    """
    # The shape of `input_0` does not change while
    # running the graph. Therefore, the static shape
    # of `input_0` is used.
    [batch_size_0, height_input_0, width_input_0, nb_maps_0] = input_0.get_shape().as_list()
    
    # The shape of `input_1` does not change while
    # running the graph. Therefore, the static shape
    # of `input_1` is used.
    [batch_size_1, height_input_1, width_input_1, nb_maps_1] = input_1.get_shape().as_list()
    nb_in_per_hidden = height_input_0*width_input_0 + height_input_1*width_input_1
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights',
                                  dtype=tf.float32,
                                  initializer=tf.random_normal([nb_maps_0, nb_in_per_hidden, height_output*width_output],
                                                               mean=0.,
                                                               stddev=1./numpy.sqrt(nb_in_per_hidden).item(),
                                                               dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 dtype=tf.float32,
                                 initializer=tf.zeros([nb_maps_0, height_output*width_output], dtype=tf.float32))
        
        transposed_input_0 = tf.transpose(tf.reshape(input_0, [batch_size_0, height_input_0*width_input_0, nb_maps_0]),
                                          [2, 0, 1])
        transposed_input_1 = tf.transpose(tf.reshape(input_1, [batch_size_1, height_input_1*width_input_1, nb_maps_1]),
                                          [2, 0, 1])
        
        # If `batch_size_0` is not equal to `batch_size_1`, `tf.concat`
        # raises a `ValueError` exception. Similarly, if `nb_maps_0` is
        # not equal to `nb_maps_1`, `tf.concat` raises a `ValueError`
        # exception.
        concatenation = tf.concat([transposed_input_0, transposed_input_1],
                                  2)
        transposed_output = tf.matmul(concatenation, weights) + tf.tile(tf.expand_dims(biases, axis=1), [1, batch_size_0, 1])
        return tf.reshape(tf.transpose(transposed_output, [1, 2, 0]),
                          [batch_size_0, height_output, width_output, nb_maps_0])

def convolution_2d_auto_width_kernels(input, nb_maps_output, stride, init_xavier, scope):
    """Convolves the input with kernels (whose width is automatically found) and adds biases.
    
    Parameters
    ----------
    input : Tensor
        4D tensor with data-type `tf.float32`.
        Input to the convolutional layer.
    nb_maps_output : int
        Number of feature maps in the output
        of the convolutional layer.
    stride : int
        Convolutional stride.
    init_xavier : bool
        Are the convolutional kernels initialized via
        Xavizer's initialization? If False, the convolutional
        kernels are initialized from a Gaussian distribution
        with mean 0 and standard deviation 0.01.
    scope : str
        Scope for the weights and biases of the convolutional layer.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output of the convolutional layer. The tensor shape
        is equal to [`input.get_shape().as_list()[0]`,
        `input.get_shape().as_list()[1]//stride`,
        `input.get_shape().as_list()[2]//stride`,
        `nb_maps_output`].
    
    """
    width_kernel = 2*stride + 1
    
    # The shape of `input` does not change while
    # running the graph. Therefore, the static shape
    # of `input` is used.
    nb_maps_input = input.get_shape().as_list()[3]
    
    # If either the input to the convolutional layer
    # or the output of the convolutional layer lives
    # in the pixel space, Xavier's initialization may
    # not be the right initialization for the convolutional
    # kernels.
    if init_xavier:
        stddev = 1./numpy.sqrt(nb_maps_input*width_kernel**2).item()
    else:
        stddev = 0.01
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights',
                                  dtype=tf.float32,
                                  initializer=tf.random_normal([width_kernel, width_kernel, nb_maps_input, nb_maps_output],
                                                                mean=0.,
                                                                stddev=stddev,
                                                                dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 dtype=tf.float32,
                                 initializer=tf.zeros([nb_maps_output], dtype=tf.float32))
        
        conv = tf.nn.conv2d(input,
                            weights,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        return tf.nn.bias_add(conv,
                              biases)

def difference_gradients_direction(tensor_0_term_minus, tensor_0_term_plus, tensor_1_term_minus, tensor_1_term_plus):
    """Computes the difference between two gradients in the same direction.
    
    The gradients are computed using finite difference.
    
    Parameters
    ----------
    tensor_0_term_minus : Tensor
        Tensor with data-type `tf.float32`.
        "minus" term in the finite difference
        for the 1st tensor.
    tensor_0_term_plus : Tensor
        Tensor with data-type `tf.float32`.
        "plus" term in the finite difference
        for the 1st tensor.
    tensor_1_term_minus : Tensor
        Tensor with data-type `tf.float32`.
        "minus" term in the finite difference
        for the 2nd tensor.
    tensor_1_term_plus : Tensor
        Tensor with data-type `tf.float32`.
        "plus" term in the finite difference
        for the 2nd tensor.
    
    Returns
    -------
    Tensor
        Tensor with data-type `tf.float32`.
        Difference between the two gradients in the
        same direction.
    
    """
    return tensor_0_term_plus - tensor_0_term_minus - (tensor_1_term_plus - tensor_1_term_minus)

def leaky_relu(input):
    """Computes Leaky ReLU with slope 0.1.
    
    Parameters
    ----------
    input : Tensor
        Tensor with data-type `tf.float32`.
        Input to Leaky ReLU.
    
    Returns
    -------
    Tensor
        Tensor with data-type `tf.float32`.
        Output of Leaky ReLU. The tensor shape
        is equal to `input.get_shape().as_list()`.
    
    """
    return tf.maximum(0.1*input, input)

def prediction_error_gradient(targets_float32, predictions_float32):
    """Computes the gradient error between the target patches and their prediction.
    
    Parameters
    ----------
    targets_float32 : Tensor
        4D tensor with data-type `tf.float32`.
        Target patches. `tf.slice(targets_float32, [i, 0, 0, 0], [1, -1, -1, -1])`
        is the target patch of index i. `targets_float32.get_shape().as_list()[3]`
        is equal to 1.
    predictions_float32 : Tensor
        4D tensor with data-type `tf.float32`.
        Prediction of the target patches. `tf.slice(predictions_float32, [i, 0, 0, 0], [1, -1, -1, -1])`
        is the prediction of the target patch of index i.
        `predictions_float32.get_shape().as_list()`
        is equal to `targets_float32.get_shape().as_list()`.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        Gradient error between the target patches and
        their prediction.
    
    """
    tuple_targets_slices = slice_4_directions(targets_float32)
    tuple_predictions_slices = slice_4_directions(predictions_float32)
    
    # If `predictions_float32.get_shape().as_list()` is not equal
    # to `targets_float32.get_shape().as_list()`, `difference_gradients_direction`
    # raises a `ValueError` exception.
    diffs_right_left = difference_gradients_direction(tuple_predictions_slices[0],
                                                      tuple_predictions_slices[1],
                                                      tuple_targets_slices[0],
                                                      tuple_targets_slices[1])
    diffs_top_right_bottom_left = difference_gradients_direction(tuple_predictions_slices[2],
                                                                 tuple_predictions_slices[3],
                                                                 tuple_targets_slices[2],
                                                                 tuple_targets_slices[3])
    diffs_top_bottom = difference_gradients_direction(tuple_predictions_slices[4],
                                                      tuple_predictions_slices[5],
                                                      tuple_targets_slices[4],
                                                      tuple_targets_slices[5])
    diffs_top_left_bottom_right = difference_gradients_direction(tuple_predictions_slices[6],
                                                                 tuple_predictions_slices[7],
                                                                 tuple_targets_slices[6],
                                                                 tuple_targets_slices[7])
    l1_norms_right_left = tf.reduce_sum(tf.abs(diffs_right_left),
                                        axis=[1, 2, 3])
    l1_norms_top_right_bottom_left = tf.reduce_sum(tf.abs(diffs_top_right_bottom_left),
                                                   axis=[1, 2, 3])
    l1_norms_top_bottom = tf.reduce_sum(tf.abs(diffs_top_bottom),
                                        axis=[1, 2, 3])
    l1_norms_top_left_bottom_right = tf.reduce_sum(tf.abs(diffs_top_left_bottom_right),
                                                   axis=[1, 2, 3])
    return tf.reduce_mean(l1_norms_right_left + l1_norms_top_right_bottom_left + l1_norms_top_bottom + l1_norms_top_left_bottom_right)

def prediction_error_l2_norm(targets_float32, predictions_float32):
    """Computes the l2-norm error between the target patches and their prediction.
    
    Parameters
    ----------
    targets_float32 : Tensor
        4D tensor with data-type `tf.float32`.
        Target patches. `tf.slice(targets_float32, [i, 0, 0, 0], [1, -1, -1, -1])`
        is the target patch of index i. `targets_float32.get_shape().as_list()[3]`
        is equal to 1.
    predictions_float32 : Tensor
        4D tensor with data-type `tf.float32`.
        Prediction of the target patches. `tf.slice(predictions_float32, [i, 0, 0, 0], [1, -1, -1, -1])`
        is the prediction of the target patch of index i.
        `predictions_float32.get_shape().as_list()`
        is equal to `targets_float32.get_shape().as_list()`.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        L2-norm error between the target patches and their prediction.
    
    """
    sums_float32 = tf.reduce_sum((targets_float32 - predictions_float32)**2,
                                 axis=[1, 2, 3])
    return tf.reduce_mean(tf.sqrt(sums_float32))

def reshape_vectors_to_channels(vectors_float32, height_channel, width_channel, name=None):
    """Reshapes each vector to an image channel.
    
    Parameters
    ----------
    vectors_float32 : Tensor
        2D tensor with data-type `tf.float32`.
        Vectors. `tf.slice(vectors_float32, [i, 0], [1, -1])`
        is the vector of index i.
    height_channel : int
        Height of the channel of each image.
    width_channel : int
        Width of the channel of each image.
    name : str, optional
        Name for the operation. The default value is None.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Images channel. The tensor shape is equal
        to [`vectors_float32.get_shape().as_list()[0]`, `height_channel`,
        `width_channel`, 1].
    
    """
    # The shape of `vectors_float32` does not change
    # while running the graph. Therefore, the
    # static shape of `vectors` is used.
    return tf.reshape(vectors_float32,
                      [vectors_float32.get_shape().as_list()[0], height_channel, width_channel, 1],
                      name=name)

def slice_4_directions(tensor_4d):
    """Slices a 4D tensor 2 times along each of the 4 directions (0, 45, 90 and 135 degrees).
    
    Parameters
    ----------
    tensor_4d : Tensor
        4D tensor with data-type `tf.float32`.
        Tensor to be sliced. `tensor_4d.get_shape().as_list()[3]`
        is equal to 1.
    
    Returns
    -------
    tuple
        Tensor
            4D tensor with data-type `tf.float32`.
            1st slice along the direction 0 degree.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            2nd slice along the direction 0 degree.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            1st slice along the direction 45 degrees.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            2nd slice along the direction 45 degrees.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            1st slice along the direction 90 degrees.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            2nd slice along the direction 90 degrees.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            1st slice along the direction 135 degrees.
            The 4th tensor dimension is equal to 1.
        Tensor
            4D tensor with data-type `tf.float32`.
            2nd slice along the direction 135 degrees.
            The 4th tensor dimension is equal to 1.
    
    """
    [batch_size, height_tensor, width_tensor, nb_maps] = tensor_4d.get_shape().as_list()
    slice_left = tf.slice(tensor_4d,
                          [0, 0, 0, 0],
                          [batch_size, height_tensor, width_tensor - 1, nb_maps])
    slice_right = tf.slice(tensor_4d,
                           [0, 0, 1, 0],
                           [batch_size, height_tensor, width_tensor - 1, nb_maps])
    slice_bottom_left = tf.slice(tensor_4d,
                                 [0, 1, 0, 0],
                                 [batch_size, height_tensor - 1, width_tensor - 1, nb_maps])
    slice_top_right = tf.slice(tensor_4d,
                               [0, 0, 1, 0],
                               [batch_size, height_tensor - 1, width_tensor - 1, nb_maps])
    slice_bottom = tf.slice(tensor_4d,
                            [0, 1, 0, 0],
                            [batch_size, height_tensor - 1, width_tensor, nb_maps])
    slice_top = tf.slice(tensor_4d,
                         [0, 0, 0, 0],
                         [batch_size, height_tensor - 1, width_tensor, nb_maps])
    slice_bottom_right = tf.slice(tensor_4d,
                                  [0, 1, 1, 0],
                                  [batch_size, height_tensor - 1, width_tensor - 1, nb_maps])
    slice_top_left = tf.slice(tensor_4d,
                              [0, 0, 0, 0],
                              [batch_size, height_tensor - 1, width_tensor - 1, nb_maps])
    
    return (
        slice_left,
        slice_right,
        slice_bottom_left,
        slice_top_right,
        slice_bottom,
        slice_top,
        slice_bottom_right,
        slice_top_left
    )

def transpose_convolution_2d_auto_width_kernels(input, nb_maps_output, stride, init_xavier, scope, name=None):
    """Transpose convolves the input with kernels (whose width is automatically found) and adds biases.
    
    Parameters
    ----------
    input : Tensor
        4D tensor with data-type `tf.float32`.
        Input to the transpose convolutional layer.
    nb_maps_output : int
        Number of feature maps in the output
        of the transpose convolutional layer.
    stride : int
        Convolutional stride.
    init_xavier : bool
        Are the convolutional kernels initialized via
        Xavizer's initialization? If False, the convolutional
        kernels are initialized from a Gaussian distribution
        with mean 0 and standard deviation 0.01.
    scope : str
        Scope for the weights and biases of the transpose convolutional layer.
    name : str, optional
        Name for the addition with biases. The default value is None.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output of the transpose convolutional layer.
        The tensor shape is equal to [`input.get_shape().as_list()[0]`,
        `input.get_shape().as_list()[1]*stride`,
        `input.get_shape().as_list()[2]*stride`,
        `nb_maps_output`].
    
    """
    width_kernel = 2*stride + 1
    
    # The shape of `input` does not change while
    # running the graph. Therefore, the static shape
    # of `input` is used.
    [batch_size, height_input, width_input, nb_maps_input] = input.get_shape().as_list()
    
    # If either the input to the transpose convolutional layer
    # or the output of the transpose convolutional layer lives
    # in the pixel space, Xavier's initialization may not be
    # the right initialization for the convolutional kernels.
    if init_xavier:
        stddev = 1./numpy.sqrt(nb_maps_input*width_kernel**2).item()
    else:
        stddev = 0.01
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights',
                                  dtype=tf.float32,
                                  initializer=tf.random_normal([width_kernel, width_kernel, nb_maps_output, nb_maps_input],
                                                                mean=0.,
                                                                stddev=stddev,
                                                                dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 dtype=tf.float32,
                                 initializer=tf.zeros([nb_maps_output], dtype=tf.float32))
        
        tconv = tf.nn.conv2d_transpose(input,
                                       weights,
                                       [batch_size, height_input*stride, width_input*stride, nb_maps_output],
                                       strides=[1, stride, stride, 1],
                                       padding='SAME')
        return tf.nn.bias_add(tconv,
                              biases,
                              name=name)


