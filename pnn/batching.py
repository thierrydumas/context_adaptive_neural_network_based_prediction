"""A library containing a function for computing a prediction of each target patch via PNN, one batch at a time."""

import numpy

import tools.tools as tls

def predict_by_batch_via_pnn(tuple_batches_float32, sess, predictor, batch_size):
    """Computes a prediction of each target patch via PNN, one batch at a time.
    
    Parameters
    ----------
    tuple_batches_float32 : tuple
        If PNN is fully-connected,
            numpy.ndarray
                2D array with data-type `numpy.float32`.
                Flattened masked contexts. `flattened_contexts_float32[i, :]`
                is the flattened masked context associated to the target
                patch of index i.
        Otherwise,
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Masked context portions, each being located
                above a different target patch. `portions_above_float32[i, :, :, :]`
                is the masked context portion located above
                the target patch of index i. `portions_above_float32.shape[3]`
                is equal to 1.
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Masked context portions, each being located
                on the left side of a different target patch.
                `portions_left_float32[i, :, :, :]` is the
                masked context portion located on the left
                side of the target patch of index i. `portions_left_float32.shape[3]`
                is equal to 1.
    sess : Session
        Session that runs the graph of PNN.
    predictor : PredictionNeuralNetwork
        PNN instance.
    batch_size : int
        Batch size.
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.float32`.
        Prediction of the target patches. The 4th
        array dimension is equal to 1.
    
    Raises
    ------
    ValueError
        If PNN is fully-connected and `numpy.sqrt(float(tuple_batches_float32[0].shape[1])/5.)`
        is not a whole number.
    
    """
    nb_predictions = tuple_batches_float32[0].shape[0]
    nb_batches = tls.divide_ints_check_divisible(nb_predictions,
                                                 batch_size)
    
    # If PNN is fully-connected, the width of the target patch
    # is retrieved from the size of its flattened masked context.
    # If PNN is convolutional, the width of the target patch is
    # equal to the height of the masked context portion located
    # above the target patch.
    if predictor.is_fully_connected:
        width_float = numpy.sqrt(float(tuple_batches_float32[0].shape[1])/5.).item()
        if not width_float.is_integer():
            raise ValueError('`numpy.sqrt(float(tuple_batches_float32[0].shape[1])/5.)` is not a whole number.')
        width_target = int(width_float)
    else:
        width_target = tuple_batches_float32[0].shape[1]
    predictions_float32 = numpy.zeros((nb_predictions, width_target, width_target, 1),
                                      dtype=numpy.float32)
    for i in range(nb_batches):
        if predictor.is_fully_connected:
            feed_dict={
                predictor.node_flattened_contexts_float32:tuple_batches_float32[0][i*batch_size:(i + 1)*batch_size, :]
            }
        else:
            feed_dict={
                predictor.node_portions_above_float32:tuple_batches_float32[0][i*batch_size:(i + 1)*batch_size, :, :, :],
                predictor.node_portions_left_float32:tuple_batches_float32[1][i*batch_size:(i + 1)*batch_size, :, :, :]
            }
        predictions_float32[i*batch_size:(i + 1)*batch_size, :, :, :] = sess.run(
            predictor.node_predictions_float32,
            feed_dict=feed_dict
        )
    return predictions_float32


