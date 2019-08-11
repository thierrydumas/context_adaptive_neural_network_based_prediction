"""A library containing functions for writing a set containing the same channel of different YCbCr images."""

import numpy
import os

# The name of the module `Queue` has been changed
# in Python 3.x <https://www.python.org/dev/peps/pep-3108/#pep-8-violations-done>.
try:
    import queue
except ImportError:
    import Queue as queue
import tensorflow as tf
import threading

import hevc.running
import sets.common
import tools.tools as tls

WIDTH_CROP = 320

# The functions are sorted in alphabetic order.

def compute_mean_intensities(paths_to_rgbs, queue_threads):
    """Computes three mean pixels intensities, each over the same channel of different YCbCr images.
    
    Parameters
    ----------
    paths_to_rgbs : list
        Each string in this list is the path to a RGB image.
    queue_threads
        Queue for storing the returned 1D array with
        data-type `numpy.float64`. The three elements
        in this array are the mean pixels intensities
        computed over the luminance channel, the blue
        chrominance channel and the red chrominance channel
        respectively of different YCbCr images.
    
    """
    accumations_float64 = numpy.zeros(3)
    nb_used = 0
    for path_to_rgb in paths_to_rgbs:
        if path_to_rgb.endswith('n02105855_2933.JPEG'):
            continue
        try:
            ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(path_to_rgb, 'RGB'))
        except (TypeError, ValueError):
            continue
        accumations_float64 += numpy.mean(ycbcr_uint8,
                                           axis=(0, 1))
        nb_used += 1
    queue_threads.put(accumations_float64/nb_used)

def compute_mean_intensities_threading(paths_to_rgbs, nb_threads):
    """Computes three mean pixels intensities, each over the same channel of different YCbCr images, via multi-threading.
    
    Parameters
    ----------
    paths_to_rgbs : list
        Each string in this list is the path to
        a RGB image. If the number of RGB images
        is very small, `nb_threads` must be equal
        to 1. Indeed, if the number of RGB images
        is very small and multi-threading is used,
        there may be imbalance in the number of RGB
        images processed by each thread. This may lead
        to some inaccuracy in the returned mean pixels
        intensities.
    nb_threads : int
        Number of threads.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.float64`.
        The three elements in this array are the mean
        pixels intensities computed over the luminance
        channel, the blue chrominance channel and the
        red chrominance channel respectively of different
        YCbCr images.
    
    """
    nb_paths_to_rgbs = len(paths_to_rgbs)
    nb_rgbs_per_thread = int(numpy.ceil(float(nb_paths_to_rgbs)/nb_threads).item())
    list_threads = []
    coordinator = tf.train.Coordinator()
    
    # The queue collects the array returned by each thread.
    # The keyworded argument `maxsize` is set for safety.
    queue_threads = queue.Queue(maxsize=nb_threads)
    for i in range(nb_threads):
        args = (
            paths_to_rgbs[i*nb_rgbs_per_thread:min(nb_paths_to_rgbs, (i + 1)*nb_rgbs_per_thread)],
            queue_threads
        )
        single_thread = threading.Thread(target=compute_mean_intensities,
                                         args=args)
        list_threads.append(single_thread)
        single_thread.start()
    coordinator.join(list_threads)
    
    # For each YCbCr image channel separately, the mean
    # pixels intensity is averaged over the different threads.
    accumulations_float64 = numpy.zeros(3)
    for _ in range(nb_threads):
        accumulations_float64 += queue_threads.get()
    return accumulations_float64/nb_threads

def count_examples_in_files_tfrecord(path_to_directory_threads):
    """Counts the total number of examples in several files ".tfrecord".
    
    Parameters
    ----------
    path_to_directory_threads : str
        Path to the directory whose subdirectories,
        named {"thread_i", i = 0 ... `nb_threads` - 1},
        store the files ".tfrecord".
    
    Returns
    -------
    int
        Total number of examples in the files ".tfrecord".
    
    """
    paths_to_tfrecords = tls.collect_paths_to_files_in_subdirectories(path_to_directory_threads,
                                                                      '.tfrecord')
    nb_examples = 0
    for path_to_tfrecord in paths_to_tfrecords:
        for _ in tf.python_io.tf_record_iterator(path_to_tfrecord):
            nb_examples += 1
    return nb_examples

def create_example_channel_single_or_pair(channel_single_or_pair_uint8):
    """Creates a Tensorflow example and fills it with the raw data bytes of the image channel.
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 1, `channel_single_or_pair_uint8` is an image channel.
        If `channel_single_or_pair_uint8.shape[2]` is equal to 2,
        `channel_single_or_pair_uint8[:, :, 0:1]` is an image
        channel and `channel_single_or_pair_uint8[:, :, 1:2]`
        is this image channel with HEVC compression artifacts.
    
    Returns
    -------
    tf.train.Example
        Tensorflow example filled with the raw data bytes
        of the image channel.
    
    Raises
    ------
    TypeError
        If `channel_single_or_pair_uint8.dtype` is not equal
        to `numpy.uint8`.
    ValueError
        If `channel_single_or_pair_uint8.ndim` is not equal to 3.
    ValueError
        `channel_single_or_pair_uint8.shape[2]` does not belong
        to {1, 2}.
    
    """
    if channel_single_or_pair_uint8.dtype != numpy.uint8:
        raise TypeError('`channel_single_or_pair_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If the check below did not exist, `create_example_channel_single_or_pair`
    # would not crash if `channel_single_or_pair_uint8.ndim` is strictly larger
    # than 3.
    if channel_single_or_pair_uint8.ndim != 3:
        raise ValueError('`channel_single_or_pair_uint8.ndim` is not equal to 3.')
    if channel_single_or_pair_uint8.shape[2] not in (1, 2):
        raise ValueError('`channel_single_or_pair_uint8.shape[2]` does not belong to {1, 2}.')
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'channel_single_or_pair': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[channel_single_or_pair_uint8.tostring()]
                    )
                )
            }
        )
    )

def create_example_context_portions_target(portion_above_uint8, portion_left_uint8, target_uint8):
    """Creates a Tensorflow example and fills it with the raw data bytes of the target patch and its two context portions.
    
    Parameters
    ----------
    portion_above_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Context portion located above the target patch.
        `portion_above_uint8.shape[2]` is equal to 1.
    portion_left_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Context portion located on the left side of the
        target patch. `portion_left_uint8.shape[2]` is
        equal to 1.
    target_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Target patch. `target_uint8.shape[2]` is equal to 1.
    
    Returns
    -------
    tf.train.Example
        Tensorflow example filled with the raw data bytes
        of the target patch and its two context portions.
    
    Raises
    ------
    TypeError
        If `portion_above_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        If `portion_left_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        If `target_uint8.dtype` is not equal to `numpy.uint8`.
    
    """
    if portion_above_uint8.dtype != numpy.uint8:
        raise TypeError('`portion_above_uint8.dtype` is not equal to `numpy.uint8`.')
    if portion_left_uint8.dtype != numpy.uint8:
        raise TypeError('`portion_left_uint8.dtype` is not equal to `numpy.uint8`.')
    if target_uint8.dtype != numpy.uint8:
        raise TypeError('`target_uint8.dtype` is not equal to `numpy.uint8`.')
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'portion_above': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[portion_above_uint8.tostring()]
                    )
                ),
                'portion_left': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[portion_left_uint8.tostring()]
                    )
                ),
                'target': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[target_uint8.tostring()]
                    )
                )
            }
        )
    )

def create_tfrecord(path_to_tfrecord, paths_to_rgbs, width_target, index_channel, dict_pair):
    """Creates a file ".tfrecord" and fills it with raw data bytes.
    
    For each RGB image, the RGB image is read, converted
    into YCbCr, and randomly cropped.
    If `width_target` is None, if `dict_pair` is None, the
    file ".tfrecord" is filled with the raw data bytes of the
    same channel from each YCbCr image. If `dict_pair` is not
    None, the file ".tfrecord" is filled with the raw data bytes
    of the same channel from each YCbCr image and this channel
    with HEVC compression artifacts.
    If `width_target` is not None, several target patches, each
    paired with its two context portions, are randomly extracted
    from the same channel from each YCbCr image. Then, the file
    ".tfrecord" is filled with the raw data bytes of each target
    patch and its two context portions. If `dict_pair` is not
    None, the context portions have HEVC compression artifacts.
    
    Parameters
    ----------
    path_to_tfrecord : str
        Path to the file ".tfrecord".
    paths_to_rgbs : list
        Each string in this list is the path to a RGB image.
    width_target : either int or None
        Width of the target patch.
    index_channel : int
        Channel index. 0 means the luminance channel.
        1 and 2 mean respectively the blue chrominance
        and the red chrominance channels.
    dict_pair : either dict or None.
        path_to_before_encoding_hevc : str
            Path to the file storing an image before the
            encoding via HEVC. `path_to_before_encoding_hevc`
            ends with ".yuv".
        path_to_after_encoding_hevc : str
            Path to the file storing the reconstructed image
            after the encoding via HEVC. `path_to_after_encoding_hevc`
            ends with ".yuv".
        path_to_cfg : str
            Path to the configuration file. `path_to_cfg` ends with ".cfg".
        path_to_bitstream : str
            Path to the bitstream file. `path_to_bitstream` ends with ".bin".
        path_to_exe_encoder : str
            Path to the executable of the HEVC encoder.
        qps_int : numpy.ndarray
            1D array whose data-type is smaller than `numpy.integer`
            in type hierarchy.
            Quantization parameters. The quantization parameter
            for encoding an image via HEVC is uniformly drawn from
            this array.
    
    Raises
    ------
    ValueError
        If `index_channel` does not belong to {0, 1, 2}.
    IOError
        If a file at `path_to_tfrecord` already exists.
    
    """
    if index_channel not in (0, 1, 2):
        raise ValueError('`index_channel` does not belong to {0, 1, 2}.')
    if os.path.isfile(path_to_tfrecord):
        raise IOError('"{}" already exists. A file ".tfrecord" cannot be overwritten.'.format(path_to_tfrecord))
    with tf.python_io.TFRecordWriter(path_to_tfrecord) as file:
        for path_to_rgb in paths_to_rgbs:
            
            # The image named "n02105855_2933.JPEG" in the ILSVRC2012 training
            # set is PNG, although its extension is not ".png".
            if path_to_rgb.endswith('n02105855_2933.JPEG'):
                continue
            try:
                ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(path_to_rgb, 'RGB'))
            except (TypeError, ValueError):
                continue
            
            # If `width_target` is None, `image_before_encoding_hevc_uint8.shape[0]`
            # and `image_before_encoding_hevc_uint8.shape[1]` have to be equal to 
            # `WIDTH_CROP`. Note that HEVC only encodes images whose height
            # and width are divisible by the minimum CU size, i.e 8 pixels.
            # Therefore, `WIDTH_CROP` has to be divisible by 8.
            # Otherwise, `image_before_encoding_hevc_uint8.shape[0]` and
            # `image_before_encoding_hevc_uint8.shape[1]` have to be divisible
            # by 8 and be larger than `3*width_target`.
            if width_target is None:
                try:
                    
                    # If `ycbcr_uint8.shape[0]` is strictly smaller
                    # than `WIDTH_CROP`, `numpy.random.choice` raises a
                    # `ValueError` exception.
                    row_start = numpy.random.choice(ycbcr_uint8.shape[0] - WIDTH_CROP + 1)
                
                    # If `ycbcr_uint8.shape[1]` is strictly smaller than
                    # `WIDTH_CROP`, `numpy.random.choice` raises a `ValueError`
                    # exception.
                    col_start = numpy.random.choice(ycbcr_uint8.shape[1] - WIDTH_CROP + 1)
                except ValueError:
                    continue
                image_before_encoding_hevc_uint8 = ycbcr_uint8[row_start:row_start + WIDTH_CROP, col_start:col_start + WIDTH_CROP, :]
            else:
                height_divisible_by_8 = 8*(ycbcr_uint8.shape[0]//8)
                width_divisible_by_8 = 8*(ycbcr_uint8.shape[1]//8)
                try:
                    
                    # If `height_divisible_by_8` is strictly smaller than
                    # `3*width_target`, `numpy.random.randint` raises a
                    # `ValueError` exception.
                    # 10 target patches, each paired with its two context
                    # portions, are extracted from each YCbCr image.
                    row_1sts = numpy.random.randint(0,
                                                    high=height_divisible_by_8 - 3*width_target + 1,
                                                    size=10)
                    
                    # If `width_divisible_by_8` is strictly smaller than
                    # `3*width_target`, `numpy.random.randint` raises a
                    # `ValueError` exception.
                    col_1sts = numpy.random.randint(0,
                                                    high=width_divisible_by_8 - 3*width_target + 1,
                                                    size=10)
                except ValueError:
                    continue
                image_before_encoding_hevc_uint8 = ycbcr_uint8[0:height_divisible_by_8, 0:width_divisible_by_8, :]
            if dict_pair is None:
                channel_single_or_pair_uint8 = image_before_encoding_hevc_uint8[:, :, index_channel:index_channel + 1]
            else:
                qp = numpy.random.choice(dict_pair['qps_int']).item()
                reconstructed_image_after_encoding_hevc_uint8 = hevc.running.encode_image(image_before_encoding_hevc_uint8,
                                                                                          dict_pair['path_to_before_encoding_hevc'],
                                                                                          dict_pair['path_to_after_encoding_hevc'],
                                                                                          dict_pair['path_to_cfg'],
                                                                                          dict_pair['path_to_bitstream'],
                                                                                          dict_pair['path_to_exe_encoder'],
                                                                                          qp,
                                                                                          None)
                
                # The channel extraction is done after the encoding
                # of the YCbCr image via HEVC.
                tuple_before_after = (
                    image_before_encoding_hevc_uint8[:, :, index_channel:index_channel + 1],
                    reconstructed_image_after_encoding_hevc_uint8[:, :, index_channel:index_channel + 1]
                )
                channel_single_or_pair_uint8 = numpy.concatenate(tuple_before_after,
                                                                 axis=2)
            if width_target is None:
                write_channel_single_or_pair(channel_single_or_pair_uint8,
                                             file)
            else:
                extract_context_portions_targets_plus_writing(channel_single_or_pair_uint8,
                                                              width_target,
                                                              row_1sts,
                                                              col_1sts,
                                                              file)

def create_tfrecord_batch(path_to_directory_tfrecords, paths_to_rgbs, width_target, index_channel,
                          nb_rgbs_per_tfrecord, dict_pair):
    """Creates a batch of files ".tfrecord" and fills the files with raw data bytes.
    
    Parameters
    ----------
    path_to_directory_tfrecords : str
        Path to the directory in which the files
        ".tfrecord" are saved.
    paths_to_rgbs : list
        Each string in this list is the path to a RGB image.
    width_target : either int or None
        Width of the target patch.
    index_channel : int
        Channel index. 0 means the luminance channel.
        1 and 2 mean respectively the blue chrominance
        and the red chrominance channels.
    nb_rgbs_per_tfrecord : int
        Maximum number of RGB images that are
        preprocessed to fill a file ".tfrecord".
    dict_pair : either dict or None.
        path_to_before_encoding_hevc : str
            Path to the file storing an image before the encoding
            via HEVC. `path_to_before_encoding_hevc` ends with ".yuv".
        path_to_after_encoding_hevc : str
            Path to the file storing the reconstructed image after
            the encoding via HEVC. `path_to_after_encoding_hevc` ends
            with ".yuv".
        path_to_cfg : str
            Path to the configuration file. `path_to_cfg` ends with ".cfg".
        path_to_bitstream : str
            Path to the bitstream file. `path_to_bitstream` ends with ".bin".
        path_to_exe_encoder : str
            Path to the executable of the HEVC encoder.
        qps_int : numpy.ndarray
            1D array whose data-type is smaller than `numpy.integer`
            in type hierarchy.
            Quantization parameters. The quantization parameter
            for encoding an image via HEVC is uniformly drawn from
            this array.
    
    """
    nb_paths_to_rgbs = len(paths_to_rgbs)
    
    # In Python 2.x, the standard division between two
    # integers returns an integer whereas, in Python 3.x,
    # the standard division between two integers returns
    # a float.
    nb_tfrecords = int(numpy.ceil(float(nb_paths_to_rgbs)/nb_rgbs_per_tfrecord).item())
    for i in range(nb_tfrecords):
        path_to_tfrecord = os.path.join(path_to_directory_tfrecords, 'data_{}.tfrecord'.format(i))
        create_tfrecord(path_to_tfrecord,
                        paths_to_rgbs[i*nb_rgbs_per_tfrecord:min(nb_paths_to_rgbs, (i + 1)*nb_rgbs_per_tfrecord)],
                        width_target,
                        index_channel,
                        dict_pair)

def create_tfrecord_threading(paths_to_directories_tfrecords, paths_to_rgbs, width_target, index_channel,
                              nb_rgbs_per_tfrecord, dict_pair_threads):
    """Creates several batches of files ".tfrecord", each batch being handled by a thread.
    
    Parameters
    ----------
    paths_to_directories_tfrecords : list
        Each string in this list is the path to a thread
        directory storing files ".tfrecord".
    paths_to_rgbs : list
        Each string in this list is the path to a RGB image.
    width_target : either int or None
        Width of the target patch.
    index_channel : int
        Channel index. 0 means the luminance channel.
        1 and 2 mean respectively the blue chrominance
        and the red chrominance channels.
    nb_rgbs_per_tfrecord : int
        Maximum number of RGB images that are
        preprocessed to fill a file ".tfrecord".
    dict_pair_threads : either dict or None.
        paths_to_before_encoding_hevc : list
            `paths_to_before_encoding_hevc[i]` is the path to
            the file storing an image before the encoding via
            HEVC for the thread of index i. Each path ends with
            ".yuv".
        paths_to_after_encoding_hevc : list
            `paths_to_after_encoding_hevc[i]` is the path to the
            file storing the reconstructed image after the encoding
            via HEVC for the thread of index i. Each path ends with
            ".yuv".
        path_to_cfg : str
            Path to the configuration file. `path_to_cfg` ends with ".cfg".
        paths_to_bitstream : list
            `paths_to_bitstream[i]` is the path to the bitstream file
            for the thread of index i. Each path ends with ".bin".
        path_to_exe_encoder : str
            Path to the executable of the HEVC encoder.
        qps_int : numpy.ndarray
            1D array whose data-type is smaller than `numpy.integer`
            in type hierarchy.
            Quantization parameters. The quantization parameter for
            encoding an image via HEVC is uniformly drawn from this
            array.
    
    """
    nb_threads = len(paths_to_directories_tfrecords)
    nb_paths_to_rgbs = len(paths_to_rgbs)
    nb_rgbs_per_thread = int(numpy.ceil(float(nb_paths_to_rgbs)/nb_threads).item())
    list_threads = []
    coordinator = tf.train.Coordinator()
    for i in range(nb_threads):
        
        # If `dict_pair_threads` is None, `dict_pair` is None too.
        if dict_pair_threads is None:
            dict_pair = None
        else:
            dict_pair = {
                'path_to_before_encoding_hevc': dict_pair_threads['paths_to_before_encoding_hevc'][i],
                'path_to_after_encoding_hevc': dict_pair_threads['paths_to_after_encoding_hevc'][i],
                'path_to_cfg': dict_pair_threads['path_to_cfg'],
                'path_to_bitstream': dict_pair_threads['paths_to_bitstream'][i],
                'path_to_exe_encoder': dict_pair_threads['path_to_exe_encoder'],
                'qps_int': dict_pair_threads['qps_int']
            }
        args = (
            paths_to_directories_tfrecords[i],
            paths_to_rgbs[i*nb_rgbs_per_thread:min(nb_paths_to_rgbs, (i + 1)*nb_rgbs_per_thread)],
            width_target,
            index_channel,
            nb_rgbs_per_tfrecord,
            dict_pair
        )
        single_thread = threading.Thread(target=create_tfrecord_batch,
                                         args=args)
        list_threads.append(single_thread)
        
        # `threading.Thread.start` calls `threading.Thread.run`
        # in a separate thread of control.
        single_thread.start()
    
    # If a thread raises an exception,
    # the coordinator blocks it until
    # the other threads finish their task.
    coordinator.join(list_threads)

def create_training_subset(paths_to_rgbs, path_to_training_subset, nb_examples):
    """Creates a subset of the set of YCbCr images that was used for creating a training set.
    
    Parameters
    ----------
    paths_to_rgbs : list
        Each string in this list is the path to a RGB image.
    path_to_training_subset : str
        Path to the file in which the training
        subset is saved. The path ends with ".npy".
    nb_examples : int
        Number of YCbCr images in the training subset.
    
    Raises
    ------
    RuntimeError
        If the training subset is not filled with `nb_examples`
        YCbCr images.
    
    """
    if os.path.isfile(path_to_training_subset):
        print('"{}" already exists.'.format(path_to_training_subset))
    else:
        ycbcrs_uint8 = numpy.zeros((nb_examples, WIDTH_CROP, WIDTH_CROP, 3),
                                   dtype=numpy.uint8)
        i = 0
        for path_to_rgb in paths_to_rgbs:
            if path_to_rgb.endswith('n02105855_2933.JPEG'):
                continue
            try:
                ycbcr_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode(path_to_rgb, 'RGB'))
            except (TypeError, ValueError):
                continue
            if ycbcr_uint8.shape[0] < WIDTH_CROP or ycbcr_uint8.shape[1] < WIDTH_CROP:
                continue
            ycbcrs_uint8[i, :, :, :] = ycbcr_uint8[0:WIDTH_CROP, 0:WIDTH_CROP, :]
            i += 1
            if i == nb_examples:
                break
        
        # If `i` is strictly smaller than `nb_examples`,
        # at least one YCbCr image in the training subset
        # is filled with 0s.
        if i != nb_examples:
            raise RuntimeError('The training subset is not filled with {} YCbCr images.'.format(nb_examples))
        numpy.save(path_to_training_subset,
                   ycbcrs_uint8)

def extract_context_portions_targets_plus_writing(channel_single_or_pair_uint8, width_target, row_1sts, col_1sts, file):
    """Extracts several target patches, each paired with its two context portions, from the image channel and writes them to the file ".tfrecord".
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 1, `channel_single_or_pair_uint8` is an image channel.
        If `channel_single_or_pair_uint8.shape[2]` is equal to 2,
        `channel_single_or_pair_uint8[:, :, 0:1]` is an image
        channel and `channel_single_or_pair_uint8[:, :, 1:2]`
        is this image channel with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    file : TFRecordWriter
        File to which each target patch and its two context portions
        are written.
    
    """
    (portions_above_uint8, portions_left_uint8, targets_uint8) = \
        sets.common.extract_context_portions_targets_from_channel_numpy(channel_single_or_pair_uint8,
                                                                        width_target,
                                                                        row_1sts,
                                                                        col_1sts)
    write_context_portions_targets(portions_above_uint8,
                                   portions_left_uint8,
                                   targets_uint8,
                                   file)

def write_channel_single_or_pair(channel_single_or_pair_uint8, file):
    """Writes the image channel to the file ".tfrecord".
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 1, `channel_single_or_pair_uint8` is an image channel.
        If `channel_single_or_pair_uint8.shape[2]` is equal to 2,
        `channel_single_or_pair_uint8[:, :, 0:1]` is an image
        channel and `channel_single_or_pair_uint8[:, :, 1:2]`
        is this image channel with HEVC compression artifacts.
    file : TFRecordWriter
        File to which the image channel is written.
    
    """
    example = create_example_channel_single_or_pair(channel_single_or_pair_uint8)
    file.write(example.SerializeToString())

def write_context_portions_targets(portions_above_uint8, portions_left_uint8, targets_uint8, file):
    """Writes each target patch and its two context portions to the file ".tfrecord".
    
    WARNING! For computation efficiency, the shape of each
    array in the function arguments is not checked.
    
    Parameters
    ----------
    portions_above_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Context portions, each being located above
        a different target patch. `portions_above_uint8[i, :, :, :]`
        is the context portion located above the
        target patch of index i. `portions_above_uint8.shape[3]`
        is equal to 1.
    portions_left_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Context portions, each being located on the
        left side of a different target patch. `portions_left_uint8[i, :, :, :]`
        is the context portion located on the left side
        of the target patch of index i. `portions_left_uint8.shape[3]`
        is equal to 1.
    targets_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Target patches. `targets_uint8[i, :, :, :]` is
        the target patch of index i. `target_uint8.shape[3]`
        is equal to 1.
    file : TFRecordWriter
        File to which each target patch and its two context
        portions are written.
    
    """
    for i in range(targets_uint8.shape[0]):
        example = create_example_context_portions_target(portions_above_uint8[i, :, :, :],
                                                         portions_left_uint8[i, :, :, :],
                                                         targets_uint8[i, :, :, :])
        file.write(example.SerializeToString())


