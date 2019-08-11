"""A library containing the components of the graph of PNN."""

import numpy
import tensorflow as tf

import pnn.tfutils

# The functions are sorted in alphabetic order.

def branch(input, strides_branch, scope):
    """Computes the output of the branch from the input to the branch.
    
    Parameters
    ----------
    input : Tensor
        4D tensor with data-type `tf.float32`.
        Input to the branch. tf.slice(input, [i, 0, 0, 0], [1, -1, -1, -1])`
        is the input example of index i. `input.get_shape().as_list()[3]`
        is equal to 1.
    strides_branch : tuple
        Each integer in this tuple is the stride of a
        different convolutional layer in the branch.
    scope : str
        Scope of the weights and biases in the branch.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output of the branch.
    
    """
    nb_maps_output = 32
    with tf.variable_scope(scope):
        for i in range(len(strides_branch)):
            nb_maps_output *= strides_branch[i]
            
            # If the input to a convolutional layer lives in the
            # pixel space, Xavier's initialization is not used.
            init_xavier = i != 0
            input = pnn.tfutils.leaky_relu(
                pnn.tfutils.convolution_2d_auto_width_kernels(input,
                                                              nb_maps_output,
                                                              strides_branch[i],
                                                              init_xavier,
                                                              'convolution_{}'.format(i))
            )
        return input

def inference_convolutional(portions_above_float32, portions_left_float32, strides_branch, scope):
    """Predicts each target patch from its two masked context portions.
    
    Parameters
    ----------
    portions_above_float32 : Tensor
        4D tensor with data-type `tf.float32`.
        Masked context portions, each being located
        above a different target patch. `tf.slice(portions_above_float32, [i ,0, 0, 0], [1, -1, -1, -1])`
        is the masked context portion located above
        the target patch of index i. `portions_above_float32.get_shape().as_list()[3]`
        is equal to 1.
    portions_left_float32 : Tensor
        4D tensor with data-type `tf.float32`.
        Masked context portions, each being located
        on the left side of a different target patch.
        `tf.slice(portions_left_float32, [i ,0, 0, 0], [1, -1, -1, -1])`
        is the masked context portion located on
        the left side of the target patch of index i.
        `portions_left_float32.get_shape().as_list()[3]`
        is equal to 1.
    strides_branch : tuple
        Each integer in this tuple is the stride of a
        different convolutional layer in a branch.
    scope : str
        Scope of the weights and biases involved in this inference.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Prediction of the target patches. The 4th
        tensor dimension is equalto 1.
    
    """
    width_target = portions_above_float32.get_shape().as_list()[1]
    with tf.variable_scope(scope):
        input_0 = branch(portions_above_float32,
                         strides_branch,
                         'branch_above')
        input_1 = branch(portions_left_float32,
                         strides_branch,
                         'branch_left')
        
        # In the merger, the series of strides is in the
        # reverse order of the series of strides in a branch.
        strides_merger = strides_branch[::-1]
        return merger(input_0,
                      input_1,
                      width_target,
                      strides_merger,
                      'merger')

def inference_fully_connected(flattened_contexts_float32, width_target, scope):
    """Predicts each target patch from its flattened masked context.
    
    Parameters
    ----------
    flattened_contexts_float32 : Tensor
        2D tensor with data-type `tf.float32`.
        Flattened masked contexts. `tf.slice(flattened_contexts_float32, [i, 0], [1, -1])`
        is the flattened masked context associated
        to the target patch of index i.
    width_target : int
        Width of the target patch.
    scope : str
        Scope of the weights and biases involved in this inference.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Prediction of the target patches. The tensor shape
        is equal to [`flattened_contexts_float32.get_shape().as_list()[0]`,
        `width_target`, `width_target`, 1].
    
    """
    with tf.variable_scope(scope):
        weights_0 = tf.get_variable('weights_0',
                                    dtype=tf.float32,
                                    initializer=tf.random_normal([5*width_target**2, 1200],
                                                                 mean=0.,
                                                                 stddev=0.01,
                                                                 dtype=tf.float32))
        biases_0 = tf.get_variable('biases_0',
                                   dtype=tf.float32,
                                   initializer=tf.zeros([1200], dtype=tf.float32))
        
        # The initialization of `weights_1` and `weights_2`
        # follows Xavier's initialization.
        weights_1 = tf.get_variable('weights_1',
                                    dtype=tf.float32,
                                    initializer=tf.random_normal([1200, 1200],
                                                                 mean=0.,
                                                                 stddev=0.029,
                                                                 dtype=tf.float32))
        biases_1 = tf.get_variable('biases_1',
                                   dtype=tf.float32,
                                   initializer=tf.zeros([1200], dtype=tf.float32))
        weights_2 = tf.get_variable('weights_2',
                                    dtype=tf.float32,
                                    initializer=tf.random_normal([1200, 1200],
                                                                 mean=0.,
                                                                 stddev=0.029,
                                                                 dtype=tf.float32))
        biases_2 = tf.get_variable('biases_2',
                                   dtype=tf.float32,
                                   initializer=tf.zeros([1200], dtype=tf.float32))
        weights_3 = tf.get_variable('weights_3',
                                    dtype=tf.float32,
                                    initializer=tf.random_normal([1200, width_target**2],
                                                                 mean=0.,
                                                                 stddev=0.01,
                                                                 dtype=tf.float32))
        biases_3 = tf.get_variable('biases_3',
                                   dtype=tf.float32,
                                   initializer=tf.zeros([width_target**2], dtype=tf.float32))
        
        # LeakyReLU is used for all non-linearities.
        leaky_relu_0 = pnn.tfutils.leaky_relu(tf.nn.bias_add(tf.matmul(flattened_contexts_float32, weights_0), biases_0))
        leaky_relu_1 = pnn.tfutils.leaky_relu(tf.nn.bias_add(tf.matmul(leaky_relu_0, weights_1), biases_1))
        leaky_relu_2 = pnn.tfutils.leaky_relu(tf.nn.bias_add(tf.matmul(leaky_relu_1, weights_2), biases_2))
        
        # Unlike the first three layers, the 4th layer
        # of the fully-connected PNN is linear.
        vectors = tf.nn.bias_add(tf.matmul(leaky_relu_2, weights_3),
                                 biases_3)
        return pnn.tfutils.reshape_vectors_to_channels(vectors,
                                                       width_target,
                                                       width_target,
                                                       name='node_output')

def merger(input_0, input_1, width_target, strides_merger, scope):
    """Computes the output of the merger from the two input stacks of feature maps.
    
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
    width_target : int
        Width of the target patch.
    strides_merger : tuple
        Each integer in this tuple is the stride of a
        different tranpose convolutional layer in the
        merger.
    scope : str
        Scope of the weights and biases in the merger.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output of the merger. The tensor shape is equal
        to [`input_0.get_shape().as_list()[0]`, `width_target`,
        `width_target`, 1].
    
    Raises
    ------
    ValueError
        If `width_target` is not divisible by the product
        of the strides.
    
    """
    prod_strides = numpy.prod(strides_merger).item()
    if width_target % prod_strides != 0:
        raise ValueError('`width_target` is not divisible by the product of the strides.')
    width_output_ch = width_target//prod_strides
    nb_strides = len(strides_merger)
    
    # The shape of `input_0` does not change while
    # running the graph. Therefore, the static shape
    # of `input_0` is used.
    nb_maps_output = input_0.get_shape().as_list()[3]
    with tf.variable_scope(scope):
        output = pnn.tfutils.leaky_relu(
            pnn.tfutils.channelwise_fully_connected_merger(input_0,
                                                           input_1,
                                                           width_output_ch,
                                                           width_output_ch,
                                                           'channelwise_fully_connected_merger')
        )
        for i in range(nb_strides):
            
            # If the output of a transpose convolutional layer
            # lives in the pixel space, Xavier's initialization
            # is not used.
            if i == nb_strides - 1:
                nb_maps_output = 1
                init_xavier = False
                name_addition_biases = 'node_output'
            else:
                nb_maps_output //= strides_merger[i]
                init_xavier = True
                name_addition_biases = None
            output = pnn.tfutils.transpose_convolution_2d_auto_width_kernels(output,
                                                                             nb_maps_output,
                                                                             strides_merger[i],
                                                                             init_xavier,
                                                                             'transpose_convolution_{}'.format(i),
                                                                             name=name_addition_biases)
            
            # The last layer of the merger is linear.
            if i != nb_strides - 1:
                output = pnn.tfutils.leaky_relu(output)
        return output

def optimizer(targets_float32, predictions_float32, learning_rate, global_step, coeff_l2_norm_pred_error, coeff_grad_error, scope):
    """Optimizes the parameters of PNN.
    
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
        is the prediction of the target patch of
        index i. `predictions_float32.get_shape().as_list()`
        is equal to `targets_float32.get_shape().as_list()`.
    learning_rate : either float or Tensor
        If Tensor, 0D tensor with data-type `tf.float32`.
        Learning rate for the parameters of PNN.
    global_step : Tensor
        0D tensor with data-type `int32_ref`.
        Global training step.
    coeff_l2_norm_pred_error : float
        Coefficient that scales the l2-norm prediction
        error with respect to the gradient error and the
        l2-norm weight decay.
    coeff_grad_error : float
        Coefficient that scales the gradient error
        with respect to the l2-norm prediction error
        and the l2-norm weight decay.
    scope : str
        Scope of the variables to be optimized.
    
    Returns
    -------
    tuple
        dict
            The two tensors in this dictionary are 0D
            tensors with data-type `tf.float32`. They
            correspond to respectively the l2-norm prediction
            error and the gradient error.
        Tensor
            0D tensor with data-type `tf.float32`.
            L2-norm weight decay.
        Operation
            Trigger for optimizing of the parameters of PNN.
    
    Raises
    ------
    ValueError
        If `coeff_l2_norm_pred_error` is not positive.
    ValueError
        If `coeff_grad_error` is not positive.
    ValueError
        If `coeff_l2_norm_pred_error` and `coeff_grad_error`
        are equal to 0.
    
    """
    if coeff_l2_norm_pred_error < 0.:
        raise ValueError('`coeff_l2_norm_pred_error` is not positive.')
    if coeff_grad_error < 0.:
        raise ValueError('`coeff_grad_error` is not positive.')
    if not coeff_l2_norm_pred_error and not coeff_grad_error:
        raise ValueError('`coeff_l2_norm_pred_error` and `coeff_grad_error` are equal to 0.')
    
    # If `coeff_l2_norm_pred_error` is equal to 0.0, we
    # should not allocate memory for the computation of
    # the l2-norm prediction error. Similarly, if `coeff_grad_error`
    # is equal to 0.0, we should not allocate memory for
    # the computation of the gradient error.
    loss_float32 = tf.constant(0.,
                               dtype=tf.float32)
    dict_errors = {}
    if coeff_l2_norm_pred_error:
        dict_errors['l2_norm_pred_error'] = coeff_l2_norm_pred_error*pnn.tfutils.prediction_error_l2_norm(targets_float32,
                                                                                                          predictions_float32)
        loss_float32 += dict_errors['l2_norm_pred_error']
    if coeff_grad_error:
        dict_errors['grad_error'] = coeff_grad_error*pnn.tfutils.prediction_error_gradient(targets_float32,
                                                                                           predictions_float32)
        loss_float32 += dict_errors['grad_error']
    weight_decay = 5.e-4*weight_l2_norm()
    loss_float32 += weight_decay
    optimization = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        loss_float32,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope),
        global_step=global_step
    )
    return (dict_errors, weight_decay, optimization)

def weight_l2_norm():
    """Computes the cumulated weight l2-norm.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        Cumulated weight l2-norm.
    
    """
    cumulated_l2_norm = tf.constant(0., dtype=tf.float32)
    for trainable_variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        name = trainable_variable.name.split('/')[-1]
        if name.startswith('weights'):
            cumulated_l2_norm += tf.nn.l2_loss(trainable_variable)
    return cumulated_l2_norm


