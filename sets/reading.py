"""A library containing functions for reading a set of images channel."""

import random
import tensorflow as tf

import sets.writing
import tools.tools as tls

# The functions are sorted in alphabetic order.

def build_batch_training(path_to_directory_threads, batch_size, nb_threads, width_target, tuple_is_extraction_is_pair,
                         mean_training, tuple_width_height_masks, is_fully_connected):
    """Builds a batch of target patches and its two batches of masked context portions for training.
    
    For training, all files ".tfrecord" are shuffled in
    the queue and all triplets {target patch, masked context
    portion located above the target patch, masked context
    portion located on the left side of the target patch}
    are shuffled for creating batches. One can cycle through
    the files ".tfrecord" indefinitely.
    
    Parameters
    ----------
    path_to_directory_threads : str
        Path to the directory whose subdirectories,
        named {"thread_i", i = 0 ... `nb_threads` - 1},
        store the files ".tfrecord".
    batch_size : int
        Batch size.
    nb_threads : int
        Number of threads.
    width_target : int
        Width of the target patch.
    tuple_is_extraction_is_pair : tuple
        bool
            Is the extraction of target patches,
            each paired with its and two context
            portions, carried out now? If False, the
            extraction was carried out when writing
            the files ".tfrecord".
        bool
            Was there a compression via HEVC when writing
            the files ".tfrecord"? This boolean can be omitted
            when the 1st boolean is False.
    mean_training : float
        Mean pixels intensity computed over the same channel
        of different YCbCr images.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the width of the mask
        that covers the right side of the context portion located
        above the target patch. The 2nd integer in this tuple is
        the height of the mask that covers the bottom of the context
        portion located on the left side of the target patch. If
        `tuple_width_height_masks` is empty, each integer is uniformly
        drawn.
    is_fully_connected : bool
        Is PNN fully-connected?
    
    Returns
    -------
    list
        If `is_fully_connected` is True,
            Tensor
                2D tensor with data-type `tf.float32`.
                Batch of flattened masked contexts. The
                tensor shape is equal to [`batch_size`,
                `5*width_target**2`].
            Tensor
                4D tensor with data-type `tf.float32`.
                Batch of target patches. The tensor shape
                is equal to [`batch_size`, `width_target`, `width_target`, 1].
        Otherwise,
            Tensor
                4D tensor with data-type `tf.float32`.
                Batch of masked context portions, each being located
                above a target patch. The tensor shape is equal to
                [`batch_size`, `width_target`, `3*width_target`, 1].
            Tensor
                4D tensor with data-type `tf.float32`.
                Batch of masked context portions, each being located
                on the left side of a target patch. The tensor shape
                is equal to [`batch_size`, `2*width_target`, `width_target`, 1].
            Tensor
                4D tensor with data-type `tf.float32`.
                Batch of target patches. The tensor shape is equal
                to [`batch_size`, `width_target`, `width_target`, 1].
    
    """
    paths_to_tfrecords = tls.collect_paths_to_files_in_subdirectories(path_to_directory_threads,
                                                                      '.tfrecord')
    random.shuffle(paths_to_tfrecords)
    with tf.device('/cpu:0'):
        
        # `tf.train.string_input_producer` creates
        # a `FIFOQueue` and adds a `QueueRunner` to
        # the collection of key "queue_runners".
        queue_tfrecord = tf.train.string_input_producer(paths_to_tfrecords)
        list_tuples = []
        for _ in range(nb_threads):
            list_tuples.append(
                read_queue_tfrecord_plus_preprocessing(queue_tfrecord,
                                                       width_target,
                                                       tuple_is_extraction_is_pair,
                                                       mean_training,
                                                       tuple_width_height_masks,
                                                       is_fully_connected)
            )
        
        # `tf.train.shuffle_batch_join` creates a
        # shuffling queue into which tensors from
        # `list_tuples` are enqueued. It also creates
        # a `dequeue_many` operation to generate
        # batches from the queue. Finally, a `QueueRunner`
        # is added to the collection of key "queue_runners".
        min_after_dequeue = 1000
        return tf.train.shuffle_batch_join(list_tuples,
                                           batch_size,
                                           min_after_dequeue + (nb_threads + 1)*batch_size,
                                           min_after_dequeue)

def extract_context_portions_target_from_channel_tf(channel_single_or_pair_uint8, width_target):
    """Extracts a target patch and its two context portions from a random position in the image channel.
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : Tensor
        3D tensor with data-type `tf.uint8`.
        If `channel_single_or_pair_uint8.get_shape().as_list()[2]`
        is equal to 1, `channel_single_or_pair_uint8` is
        an image channel. If `channel_single_or_pair_uint8.get_shape().as_list()[2]`
        is equal to 2, `tf.slice(channel_single_or_pair_uint8, [0, 0, 0], [-1, -1, 1])`
        is an image channel and `tf.slice(channel_single_or_pair_uint8, [0, 0, 1], [-1, -1, 1])`
        is this channel with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    
    Returns
    -------
    tuple
        Tensor
            3D tensor with data-type `tf.uint8`.
            Context portion located above the target patch.
            The tensor shape is equal to [`width_target`,
            `3*width_target`, 1]. If `channel_single_or_pair_uint8.get_shape().as_list()[2]`
            is equal to 1, the context portion located above
            the target patch is extracted from the image channel.
            If `channel_single_or_pair_uint8.get_shape().as_list()[2]`
            is equal to 2, it is extracted from the image channel
            with HEVC compression artifacts.
        Tensor
            3D tensor with data-type `tf.uint8`.
            Context portion located on the left side of the
            target patch. The tensor shape is equal to
            [`2*width_target`, `width_target`, 1].  If
            `channel_single_or_pair_uint8.get_shape().as_list()[2]`
            is equal to 1, the context portion located
            on the left side of the target patch is extracted
            from the image channel. If `channel_single_or_pair_uint8.get_shape().as_list()[2]`
            is equal to 2, it is extracted from the image channel
            with HEVC compression artifacts.
        Tensor
            3D tensor with data-type `tf.uint8`.
            Target patch. The tensor shape is equal to [`width_target`,
            `width_target`, 1]. The target patch is extracted from the
            image channel.
    
    Raises
    ------
    ValueError
        If `width_target` is not positive.
    ValueError
        `channel_single_or_pair_uint8.get_shape().as_list()[2]`
        does not belong to {1, 2}.
    
    """
    rotated_channel_uint8 = rotate_randomly_0_90_180_270(channel_single_or_pair_uint8)
    
    # `rotated_channel_uint8` is flipped with probability 0.5.
    flipped_channel_uint8 = tf.image.random_flip_left_right(rotated_channel_uint8)
    
    # The shape of `flipped_channel_uint8` does not
    # change while running the graph. Therefore, the
    # static shape of `flipped_channel_uint8` is used.
    [height_flipped_channel, width_flipped_channel, nb_channels] = flipped_channel_uint8.get_shape().as_list()
    if width_target < 0:
        raise ValueError('`width_target` is not positive.')
    row_1st = tf.random_uniform([],
                                minval=0,
                                maxval=height_flipped_channel - 3*width_target + 1,
                                dtype=tf.int32)
    col_1st = tf.random_uniform([],
                                minval=0,
                                maxval=width_flipped_channel - 3*width_target + 1,
                                dtype=tf.int32)
    if nb_channels in (1, 2):
        i = nb_channels - 1
    else:
        raise ValueError('`channel_single_or_pair_uint8.get_shape().as_list()[2]` does not belong to {1, 2}.')
    portion_above_uint8 = tf.slice(flipped_channel_uint8,
                                   [row_1st, col_1st, i],
                                   [width_target, 3*width_target, 1])
    portion_left_uint8 = tf.slice(flipped_channel_uint8,
                                  [row_1st + width_target, col_1st, i],
                                  [2*width_target, width_target, 1])
    target_uint8 = tf.slice(flipped_channel_uint8,
                            [row_1st + width_target, col_1st + width_target, 0],
                            [width_target, width_target, 1])
    return (portion_above_uint8, portion_left_uint8, target_uint8)

def parse_example_channel_single_or_pair(serialized_example, is_pair):
    """Parses a serialized example containing the raw data bytes of an image channel.
    
    Parameters
    ----------
    serialized_example : Tensor
        0D tensor with data-type `tf.string`.
        Tensor containing the serialized example.
    is_pair : bool
        Does the serialized example contain an image
        channel and this channel with HEVC compression
        artifacts? If False, the serialized example
        only contains the image channel.
    
    Returns
    -------
    Tensor
        3D tensor with data-type `tf.uint8`.
        If `is_pair` is True, the tensor contains the
        image channel and this channel with HEVC compression
        artifacts. The tensor shape is equal to [`sets.writing.WIDTH_CROP`,
        `sets.writing.WIDTH_CROP`, 2]. If `is_pair` is False,
        the tensor contains the image channel. The tensor shape
        is equal to [`sets.writing.WIDTH_CROP`, `sets.writing.WIDTH_CROP`, 1].
    
    """
    dictionary_features = tf.parse_single_example(
        serialized_example,
        features={
            'channel_single_or_pair': tf.FixedLenFeature([], tf.string)
        }
    )
    flattened_channel_single_or_pair = tf.decode_raw(dictionary_features['channel_single_or_pair'],
                                                     tf.uint8)
    if is_pair:
        nb_channels = 2
    else:
        nb_channels = 1
    return tf.reshape(flattened_channel_single_or_pair,
                      [sets.writing.WIDTH_CROP, sets.writing.WIDTH_CROP, nb_channels])

def parse_example_context_portions_target(serialized_example, width_target):
    """Parses a serialized example containing the raw data bytes of a target patch and its two context portions.
    
    Parameters
    ----------
    serialized_example : Tensor
        0D tensor with data-type `tf.string`.
        Tensor containing the serialized example.
    width_target : int
        Width of the target patch.
    
    Returns
    -------
    tuple
        Tensor
            3D tensor with data-type `tf.uint8`.
            Context portion located above the target patch.
            The tensor shape is equal to [`width_target`,
            `3*width_target`, 1].
        Tensor
            3D tensor with data-type `tf.uint8`.
            Context portion located on the left side of the
            target patch. The tensor shape is equal to
            [`2*width_target`, `width_target`, 1].
        Tensor
            3D tensor with data-type `tf.uint8`.
            Target patch. The tensor shape is equal to
            [`width_target`, `width_target`, 1].
    
    """
    dictionary_features = tf.parse_single_example(
        serialized_example,
        features={
            'portion_above': tf.FixedLenFeature([], tf.string),
            'portion_left': tf.FixedLenFeature([], tf.string),
            'target': tf.FixedLenFeature([], tf.string)
        }
    )
    flattened_portion_above_uint8 = tf.decode_raw(dictionary_features['portion_above'],
                                                  tf.uint8)
    flattened_portion_left_uint8 = tf.decode_raw(dictionary_features['portion_left'],
                                                  tf.uint8)
    flattened_target_uint8 = tf.decode_raw(dictionary_features['target'],
                                           tf.uint8)
    portion_above_uint8 = tf.reshape(flattened_portion_above_uint8,
                                     [width_target, 3*width_target, 1])
    portion_left_uint8 = tf.reshape(flattened_portion_left_uint8,
                                    [2*width_target, width_target, 1])
    target_uint8 = tf.reshape(flattened_target_uint8,
                              [width_target, width_target, 1])
    return (portion_above_uint8, portion_left_uint8, target_uint8)

def preprocess_context_portions_target_tf(portion_above_uint8, portion_left_uint8, target_uint8, mean_training,
                                          tuple_width_height_masks, is_fully_connected):
    """Shifts the pixel intensity in each target patch and its two context portions, then masks the two context portions.
    
    A mask covers the right side of the context portion
    located above the target patch. Its height is equal
    to `target_uint8.get_shape().as_list()[0]` and its
    width is either fixed or uniformly drawn. Another mask
    covers the bottom of the context portion located on the
    left side of the target patch. Its height is either fixed
    or uniformly drawn and its width is equal to `target_uint8.get_shape().as_list()[0]`.
    
    WARNING! The data-type and the shape of each tensor
    in the function arguments are not checked as these
    tensors come from `read_queue_tfrecord`.
    
    Parameters
    ----------
    portion_above_uint8 : tensor
        3D tensor with data-type `tf.uint8`.
        Context portion located above the target patch.
        `portion_above_uint8.get_shape().as_list()[2]`
        is equal to 1.
    portion_left_uint8 : tensor
        3D tensor with data-type `tf.uint8`.
        Context portion located on the left side of
        the target patch. `portion_left_uint8.get_shape().as_list()[2]`
        is equal to 1.
    target_uint8 : tensor
        3D tensor with data-type `tf.uint8`.
        Target patch. `target_uint8.get_shape().as_list()[2]`
        is equal to 1.
    mean_training : float
        Mean pixels intensity computed over the same channel
        of different YCbCr images.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the width of the mask
        that covers the right side of the context portion located
        above the target patch. The 2nd integer in this tuple is
        the height of the mask that covers the bottom of the context
        portion located on the left side of the target patch. If
        `tuple_width_height_masks` is empty, each integer is uniformly
        drawn.
    is_fully_connected : bool
        Is PNN fully-connected?
    
    Returns
    -------
    tuple
        If `is_fully_connected` is True,
            Tensor
                1D tensor with data-type `tf.float32`.
                Flattened masked context.
            Tensor
                3D tensor with data-type `tf.float32`.
                Target patch. The 3rd tensor dimension is equal to 1.
        Otherwise,
            Tensor
                3D tensor with data-type `tf.float32`.
                Masked context portion located above the target
                patch. The 3rd tensor dimension is equal to 1.
            Tensor
                3D tensor with data-type `tf.float32`.
                Masked context portion located on the left side
                of the target patch. The 3rd tensor dimension is
                equal to 1.
            Tensor
                3D tensor with data-type `tf.float32`.
                Target patch. The 3rd tensor dimension is equal to 1.
    
    Raises
    ------
    ValueError
        If `target_uint8.get_shape().as_list()[0]` is not
        divisible by 4.
    ValueError
        If `len(tuple_width_height_masks)` is equal to 2 and
        `tuple_width_height_masks[0]` does not belong to
        {0, 4, ..., `target_uint8.get_shape().as_list()[0]`}.
    ValueError
        If `len(tuple_width_height_masks)` is equal to 2 and
        `tuple_width_height_masks[1]` does not belong to
        {0, 4, ..., `target_uint8.get_shape().as_list()[0]`}.
    
    """
    width_target = target_uint8.get_shape().as_list()[0]
    if width_target % 4 != 0:
        raise ValueError('`target_uint8.get_shape().as_list()[0]` is not divisible by 4.')
    if tuple_width_height_masks:
        
        # If `len(tuple_width_height_masks)` is not equal to 2,
        # the unpacking below raises a `ValueError` exception.
        (width_mask_above, height_mask_left) = tuple_width_height_masks
        
        # In Tensorflow, wacky slicings do not raise any exception.
        if width_mask_above < 0 or width_mask_above > width_target or width_mask_above % 4 != 0:
            raise ValueError('`tuple_width_height_masks[0]` does not belong to {0, 4, ..., `target_uint8.get_shape().as_list()[0]`}.')
        if height_mask_left < 0 or height_mask_left > width_target or height_mask_left % 4 != 0:
            raise ValueError('`tuple_width_height_masks[1]` does not belong to {0, 4, ..., `target_uint8.get_shape().as_list()[0]`}.')
    else:
        width_mask_above = 4*tf.random_uniform([],
                                               minval=0,
                                               maxval=width_target//4 + 1,
                                               dtype=tf.int32)
        height_mask_left = 4*tf.random_uniform([],
                                               minval=0,
                                               maxval=width_target//4 + 1,
                                               dtype=tf.int32)
    
    # `sub_above_float32` is the context portion
    # located above the target patch after subtracting
    # `mean_training`.
    sub_above_float32 = tf.cast(portion_above_uint8, tf.float32) - mean_training
    indices_cols = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, 3*width_target), axis=0), axis=2),
                           [width_target, 1, 1])
    bools_0 = tf.less_equal(3*width_target - width_mask_above,
                            indices_cols)
    portion_above_float32 = tf.where(bools_0,
                                     x=tf.zeros([width_target, 3*width_target, 1], dtype=tf.float32),
                                     y=sub_above_float32)
    
    # `sub_left_float32` is the context portion
    # located on the left side of the target patch
    # after subtracting `mean_training`.
    sub_left_float32 = tf.cast(portion_left_uint8, tf.float32) - mean_training
    indices_rows = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, 2*width_target), axis=1), axis=2),
                           [1, width_target, 1])
    bools_1 = tf.less_equal(2*width_target - height_mask_left,
                            indices_rows)
    portion_left_float32 = tf.where(bools_1,
                                    x=tf.zeros([2*width_target, width_target, 1], dtype=tf.float32),
                                    y=sub_left_float32)
    
    # `target_float32` is the target patch after
    # subtracting `mean_training`.
    target_float32 = tf.cast(target_uint8, tf.float32) - mean_training
    if is_fully_connected:
        flattened_context_float32 = tf.concat([tf.reshape(portion_above_float32, [-1]), tf.reshape(portion_left_float32, [-1])], 0)
        return (flattened_context_float32, target_float32)
    else:
        return (portion_above_float32, portion_left_float32, target_float32)

def read_queue_tfrecord(queue_tfrecord, width_target, tuple_is_extraction_is_pair):
    """Reads a queue of files ".tfrecord" using a new reader.
    
    Parameters
    ----------
    queue_tfrecord : FIFOQueue
        Queue of files ".tfrecord".
    width_target : int
        Width of the target patch.
    tuple_is_extraction_is_pair : tuple
        bool
            Is the extraction of target patches,
            each paired with its and two context
            portions, carried out now? If False, the
            extraction was carried out when writing
            the files ".tfrecord".
        bool
            Was there a compression via HEVC when
            writing the files ".tfrecord"? This boolean
            can be omitted when the 1st boolean is False.
    
    Returns
    -------
    tuple
        Tensor
            3D tensor with data-type `tf.uint8`.
            Context portion located above the target patch.
            The tensor shape is equal to [`width_target`,
            `3*width_target`, 1].
        Tensor
            3D tensor with data-type `tf.uint8`.
            Context portion located on the left side of the
            target patch. The tensor shape is equal to
            [`2*width_target`, `width_target`, 1].
        Tensor
            3D tensor with data-type `tf.uint8`.
            Target patch. The tensor shape is equal to
            [`width_target`, `width_target`, 1].
    
    """
    reader = tf.TFRecordReader()
    serialized_example = reader.read(queue_tfrecord)[1]
    if tuple_is_extraction_is_pair[0]:
        channel_single_or_pair_uint8 = parse_example_channel_single_or_pair(serialized_example,
                                                                            tuple_is_extraction_is_pair[1])
        return extract_context_portions_target_from_channel_tf(channel_single_or_pair_uint8,
                                                               width_target)
    else:
        return parse_example_context_portions_target(serialized_example,
                                                     width_target)
                                                     
def read_queue_tfrecord_plus_preprocessing(queue_tfrecord, width_target, tuple_is_extraction_is_pair,
                                           mean_training, tuple_width_height_masks, is_fully_connected):
    """Reads a queue of files ".tfrecord" using a new reader and preprocesses the data from these files.
    
    Parameters
    ----------
    queue_tfrecord : FIFOQueue
        Queue of files ".tfrecord".
    width_target : int
        Width of the target patch.
    tuple_is_extraction_is_pair : tuple
        bool
            Is the extraction of target patches,
            each paired with its and two context
            portions, carried out now? If False, the
            extraction was carried out when writing
            the files ".tfrecord".
        bool
            Was there a compression via HEVC when
            writing the files ".tfrecord"? This boolean
            can be omitted when the 1st boolean is False.
    mean_training : float
        Mean pixels intensity computed over the same channel
        of different YCbCr images.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the width of the mask
        that covers the right side of the context portion located
        above the target patch. The 2nd integer in this tuple is
        the height of the mask that covers the bottom of the context
        portion located on the left side of the target patch. If
        `tuple_width_height_masks` is empty, each integer is uniformly
        drawn.
    is_fully_connected : bool
        Is PNN fully-connected?
    
    Returns
    -------
    tuple
        If `is_fully_connected` is True,
            Tensor
                1D tensor with data-type `tf.float32`.
                Flattened masked context. The tensor shape
                is equal to [`5*width_target**2`,].
            Tensor
                3D tensor with data-type `tf.float32`.
                Target patch. The tensor shape is equal to
                [`width_target`, `width_target`, 1].
        Otherwise,
            Tensor
                3D tensor with data-type `tf.float32`.
                Masked context portion located above the target
                patch. The tensor shape is equal to [`width_target`,
                `3*width_target`, 1].
            Tensor
                3D tensor with data-type `tf.float32`.
                Masked context portion located on the left side
                of the target patch. The tensor shape is equal to
                [`2*width_target`, `width_target`, 1].
            Tensor
                3D tensor with data-type `tf.float32`.
                Target patch. The tensor shape is equal to
                [`width_target`, `width_target`, 1].
    
    """
    (portion_above_uint8, portion_left_uint8, target_uint8) = read_queue_tfrecord(queue_tfrecord,
                                                                                  width_target,
                                                                                  tuple_is_extraction_is_pair)
    return preprocess_context_portions_target_tf(portion_above_uint8,
                                                 portion_left_uint8,
                                                 target_uint8,
                                                 mean_training,
                                                 tuple_width_height_masks,
                                                 is_fully_connected)

def rotate_randomly_0_90_180_270(input):
    """Rotates the input tensor by an angle uniformly drawn from {0, 90, 180, 270} degrees.
    
    Parameters
    ----------
    input : Tensor
        3D tensor.
        Input tensor to be rotated.
    
    Returns
    -------
    Tensor
        Tensor with the same number of dimensions
        and the same data-type as `input`.
        Rotated input.
    
    """
    k = tf.random_uniform([],
                          minval=0,
                          maxval=4,
                          dtype=tf.int32)
    return tf.image.rot90(input,
                          k=k)


