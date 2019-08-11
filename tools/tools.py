"""A library containing generic functions."""

import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy
import os
import PIL.Image

# The functions are sorted in alphabetic order.

def cast_float_to_uint8(array_float):
    """Casts the array elements from float to 8-bit unsigned integer.
    
    The array elements are clipped to [0., 255.],
    rounded to the nearest whole number, and cast
    from float to 8-bit unsigned integer.
    
    Parameters
    ----------
    array_float : numpy.ndarray
        Array whose data-type is smaller than
        `numpy.float` in type hierarchy.
    
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.uint8`. The
        array has the same shape as `array_float`.
    
    Raises
    ------
    TypeError
        If `array_float.dtype` is not smaller than
        `numpy.float` in type hierarchy.
    
    """
    if not numpy.issubdtype(array_float.dtype, numpy.float):
        raise TypeError('`array_float.dtype` is not smaller than `numpy.float` in type hierarchy.')
    
    # If the above check did not exist and `array_float`
    # contained integers, the clipping below would cast
    # the integers to floats.
    return numpy.round(array_float.clip(min=0., max=255.)).astype(numpy.uint8)

def ceil_float(input_float, nb_digits):
    """Ceils the input float to `nb_digits` digits from the decimal point.
    
    Parameters
    ----------
    input_float : float
        Input float to be ceiled.
    nb_digits : int
        Number of digits from the decimal point.
    
    Returns
    -------
    float
        Result of the ceiling.
    
    Raises
    ------
    TypeError
        If `input_float` is not an instance of `float`.
    TypeError
        If `nb_digits` is not an instance of `int`.
    
    """
    if not isinstance(input_float, float):
        raise TypeError('`input_float` is not an instance of `float`.')
    if not isinstance(nb_digits, int):
        raise TypeError('`nb_digits` is not an instance of `int`.')
    factor = 10**nb_digits
    return numpy.ceil(input_float*factor).item()/factor

def check_remove_zero_slices_in_array_3d(array_3d, are_zero_slices):
    """Checks whether the zero slices in a 3D array match the expected zero slices and returns a new array without these slices.
    
    Parameters
    ----------
    array_3d : numpy.ndarray
        3D array.
        Array whose zero slices are to be removed.
    are_zero_slices : numpy.ndarray
        1D array with data-type `numpy.bool`.
        `are_zero_slices[i]` indicates whether `array_3d[:, :, i]`
        is expected to be a zero slice.
    
    Returns
    -------
    numpy.ndarray
        3D array with the same data-type as `array_3d`.
        Array without the zero slices.
    
    Raises
    ------
    ValueError
        If `array_3d.ndim` is not equal to 3.
    AssertionError
        If the found zero slices in the 3D array do not
        match the expected zero slices.
    
    """
    if array_3d.ndim != 3:
        raise ValueError('`array_3d.ndim` is not equal to 3.')
    found_zero_slices = numpy.logical_not(array_3d.any(axis=(0, 1)))
    numpy.testing.assert_equal(found_zero_slices,
                               are_zero_slices,
                               err_msg='The found zero slices in the 3D array do not match the expected zero slices.')
    indices_int64 = numpy.where(found_zero_slices)[0]
    return numpy.delete(array_3d,
                        indices_int64,
                        axis=2)

def clean_sort_list_strings(list_strings, extension):
    """Removes from the list the strings that do not end with the given extensions and sorts the list.
    
    Parameters
    ----------
    list_strings : list
        List of strings.
    extension : str or tuple of strs
        Given extensions.
    
    Returns
    -------
    list
        New list which contains the strings that
        end with the given extensions. This list
        is sorted.
    
    """
    list_strings_extension = [string for string in list_strings if string.endswith(extension)]
    list_strings_extension.sort()
    return list_strings_extension

def collect_integer_between_tags_in_each_filename(path_to_directory, extensions, tag_0, tag_1):
    """Collects the integer between two given tags in each filename with a given extension.
    
    Parameters
    ----------
    path_to_directory : str
        Path to the directory containing the
        files to be considered.
    extensions : tuple
        Each string in this tuple is a possible
        extension of the files to be considered.
    tag_0 : str
        1st given tag.
    tag_1 : str
        2nd given tag.
    
    Returns
    -------
    list
        Each integer in this list is the integer
        found between the two given tags in a filename
        with a given extension. The list is sorted
        in ascending order.
    
    """
    list_integers = []
    for name_item in os.listdir(path_to_directory):
        if name_item.endswith(extensions):
            index_start_tag_0 = name_item.find(tag_0)
            index_start_tag_1 = name_item.find(tag_1)
            if index_start_tag_0 == -1 or index_start_tag_1 == -1:
                continue
            else:
                try:
                    substring = name_item[index_start_tag_0 + len(tag_0):index_start_tag_1]
                    list_integers.append(int(substring))
                
                # If `substring` cannot be converted into
                # an integer, the filename `name_item` is
                # ignored.
                except ValueError:
                    continue
    list_integers.sort()
    return list_integers

def collect_paths_to_files_in_subdirectories(path_to_directory, extension):
    """Collects the path to each file stored in the subdirectories of a given directory.
    
    If the extension of a file does not match the given
    extension, the path to the file is not collected.
    
    Parameters
    ----------
    path_to_directory : str
        Path to the given directory.
    extension : either str or tuple of strs
        Given file extension(s).
    
    Returns
    -------
    list
        Each string in this list is the path to
        a file stored in the subdirectories of the
        given directory. This list is sorted.
    
    """
    paths_to_subdirectories = collect_paths_to_subdirectories(path_to_directory)
    paths_to_files = []
    for path_to_subdirectory in paths_to_subdirectories:
        for name_item in os.listdir(path_to_subdirectory):
            path_to_item = os.path.join(path_to_subdirectory,
                                        name_item)
            
            # A subdirectory whose name ends with the given
            # extension is not added to `paths_to_files`.
            if path_to_item.endswith(extension) and not os.path.isdir(path_to_item):
                paths_to_files.append(path_to_item)
    
    # `os.listdir` returns a list whose order depends
    # on the OS. To make `collect_paths_to_files_in_subdirectories`
    # independent of the OS, `paths_to_files` is sorted.
    paths_to_files.sort()
    return paths_to_files

def collect_paths_to_subdirectories(path_to_directory):
    """Collects the path to each subdirectory of a given directory.
    
    Parameters
    ----------
    path_to_directory : str
        Path to the given directory.
    
    Returns
    -------
    list
        Each string in this list is the path to
        a subdirectory of the given directory.
        This list is sorted.
    
    """
    paths_to_subdirectories = []
    for name_item in os.listdir(path_to_directory):
        path_to_item = os.path.join(path_to_directory,
                                    name_item)
        if os.path.isdir(path_to_item):
            paths_to_subdirectories.append(path_to_item)
    
    # `os.listdir` returns a list whose order depends
    # on the OS. To make `collect_paths_to_subdirectories`
    # independent of the OS, `paths_to_subdirectories` is
    # sorted.
    paths_to_subdirectories.sort()
    return paths_to_subdirectories

def compute_bjontegaard(rates_0, psnrs_0, rates_1, psnrs_1):
    """Compute Bjontegaard's metric between two rate-distortion curves.
    
    Bjontegaard's metric is the average per cent saving in
    bitrate between two rate-distortion curves.
    
    Parameters
    ----------
    rates_0 : numpy.ndarray
        1D array.
        Rates for the 1st rate-distortion curve. `rates_0[i]`
        is the rate of the point of index i in the 1st rate-distortion
        curve.
    psnrs_0 : numpy.ndarray
        1D array.
        PSNRs for the 1st rate-distortion curve. `psnrs_0[i]`
        is the PSNR of the point of index i in the 1st rate-distortion
        curve.
    rates_1 : numpy.ndarray
        1D array.
        Rates for the 2nd rate-distortion curve. `rates_1[i]`
        is the rate of the point of index i in the 2nd rate-distortion
        curve.
    psnrs_1 : numpy.ndarray
        1D array.
        PSNRs for the 2nd rate-distortion curve. `psnrs_1[i]`
        is the PSNR of the point of index i in the 2nd rate-distortion
        curve.
    
    Returns
    -------
    float
        Bjontegaard's metric between the two rate-distortion curves.
    
    Raises
    ------
    ValueError
        If rates_0.ndim` is not equal to 1.
    ValueError
        If `rates_1.ndim` is not equal to 1.
    ValueError
        If `psnrs_0.shape` is not equal to `rates_0.shape`.
    ValueError
        If `psnrs_1.shape` is not equal to `rates_1.shape`.
    AssertionError
        If an element of `rates_0` is not strictly positive.
    AssertionError
        If an element of `rates_1` is not strictly positive.
    AssertionError
        If an element of `psnrs_0` is not strictly positive.
    AssertionError
        If an element of `psnrs_1` is not strictly positive.
    
    """
    if rates_0.ndim != 1:
        raise ValueError('`rates_0.ndim` is not equal to 1.')
    if rates_1.ndim != 1:
        raise ValueError('`rates_1.ndim` is not equal to 1.')
    if psnrs_0.shape != rates_0.shape:
        raise ValueError('`psnrs_0.shape` is not equal to `rates_0.shape`.')
    if psnrs_1.shape != rates_1.shape:
        raise ValueError('`psnrs_1.shape` is not equal to `rates_1.shape`.')
    numpy.testing.assert_array_less(0.,
                                    rates_0,
                                    err_msg='An element of `rates_0` is not strictly positive.')
    numpy.testing.assert_array_less(0.,
                                    rates_1,
                                    err_msg='An element of `rates_1` is not strictly positive.')
    numpy.testing.assert_array_less(0.,
                                    psnrs_0,
                                    err_msg='An element of `psnrs_0` is not strictly positive.')
    numpy.testing.assert_array_less(0.,
                                    psnrs_1,
                                    err_msg='An element of `psnrs_1` is not strictly positive.')
    
    # Rates are converted into logarithmic units.
    log_rates_0 = numpy.log(rates_0)
    log_rates_1 = numpy.log(rates_1)
    
    # A polynomial of degree 3 is fitted to the points
    # (`psnrs_0`, `log_rates_0`).
    polynomial_coefficients_0 = numpy.polyfit(psnrs_0,
                                              log_rates_0,
                                              3)
    
    # A polynomial of degree 3 is fitted to the points
    # (`psnrs_1`, `log_rates_1`).
    polynomial_coefficients_1 = numpy.polyfit(psnrs_1,
                                              log_rates_1,
                                              3)
    minimum = max(numpy.amin(psnrs_0).item(),
                  numpy.amin(psnrs_1).item())
    maximum = min(numpy.amax(psnrs_0).item(),
                  numpy.amax(psnrs_1).item())
    
    # `antiderivative_0` is the antiderivative (indefinite
    # integral) of the polynomial with polynomial coefficients
    # `polynomial_coefficients_0`.
    antiderivative_0 = numpy.polyint(polynomial_coefficients_0)
    
    # `antiderivative_1` is the antiderivative (indefinite
    # integral) of the polynomial with polynomial coefficients
    # `polynomial_coefficients_1`.
    antiderivative_1 = numpy.polyint(polynomial_coefficients_1)
    integral_0 = numpy.polyval(antiderivative_0, maximum) - numpy.polyval(antiderivative_0, minimum)
    integral_1 = numpy.polyval(antiderivative_1, maximum) - numpy.polyval(antiderivative_1, minimum)
    return 100.*(numpy.exp((integral_1 - integral_0)/(maximum - minimum)).item() - 1.)

def compute_psnr(array_0_uint8, array_1_uint8):
    """Computes the PSNR between the two arrays.
    
    Parameters
    ----------
    array_0_uint8 : numpy.ndarray
        Array with data-type `numpy.uint8`.
        1st array.
    array_1_uint8 : numpy.ndarray
        Array with data-type `numpy.uint8`.
        2nd array.
    
    Returns
    -------
    numpy.float64
        PSNR between the two arrays.
    
    Raises
    ------
    TypeError
        If `array_0_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        If `array_1_uint8.dtype` is not equal to `numpy.uint8`.
    
    """
    if array_0_uint8.dtype != numpy.uint8:
        raise TypeError('`array_0_uint8.dtype` is not equal to `numpy.uint8`.')
    if array_1_uint8.dtype != numpy.uint8:
        raise TypeError('`array_1_uint8.dtype` is not equal to `numpy.uint8`.')
    array_0_float64 = array_0_uint8.astype(numpy.float64)
    array_1_float64 = array_1_uint8.astype(numpy.float64)
    mse_float64 = numpy.mean((array_0_float64 - array_1_float64)**2)
    
    # `array_0_float64` and `array_1_float64` might be identical.
    # 1.e-6 is added to `mse_float64` to avoid dividing by 0.
    # The precedence of ...**... (exponentiation) is higher
    # than the precedence of .../... (division).
    return 10.*numpy.log10(255.**2/(mse_float64 + 1.e-6))

def divide_ints_check_divisible(numerator, denominator):
    """Divides the numerator by the denominator while checking that the numerator is divisible by the denominator.
    
    Parameters
    ----------
    numerator : int
        Numerator.
    denominator : int
        Denominator.
    
    Returns
    -------
    int
        Result of the division.
    
    Raises
    ------
    TypeError
        If `numerator` is not an instance of `int`.
    TypeError
        If `denominator` is not an instance of `int`.
    ValueError    
        If `numerator` is not divisible by `denominator`.
    
    """
    if not isinstance(numerator, int):
        raise TypeError('`numerator` is not an instance of `int`.')
    if not isinstance(denominator, int):
        raise TypeError('`denominator` is not an instance of `int`.')
    if numerator % denominator != 0:
        raise ValueError('`numerator` is not divisible by `denominator`.')
    return numerator//denominator

def float_to_str(float_in):
    """Converts the float into a string.
    
    During the conversion, "." is replaced
    by "dot" if the float is not a whole
    number and "-" is replaced by "minus".
    
    Parameters
    ----------
    float_in : float
        Float to be converted.
    
    Returns
    -------
    str
        String resulting from
        the conversion.
    
    """
    if float_in.is_integer():
        str_in = str(int(float_in))
    else:
        str_in = str(float_in).replace('.', 'dot')
    return str_in.replace('-', 'minus')

def histogram(data, title, path):
    """Creates a histogram of the data and saves the histogram.
    
    Parameters
    ----------
    data : numpy.ndarray
        1D array.
        Data.
    title : str
        Title of the histogram.
    path : str
        Path to the saved histogram. The path ends
        with ".png".
    
    """
    plt.hist(data,
             bins=60)
    plt.xticks(size=22)
    plt.yticks(size=22)
    plt.title(title,
              fontsize=30)
    plt.savefig(path)
    plt.clf()

def plot_bars_yaxis_limits(x_values, y_values, tuple_yaxis_limits, title, path):
    """Plots bars while imposing the y-axis lower and upper limits and saves the plot.
    
    Parameters
    ----------
    x_values : numpy.ndarray
        1D array.
        x-axis values.
    y_values : numpy.ndarray
        1D array.
        y-axis values.
    tuple_yaxis_limits : tuple
        The two elements in this tuple are the y-axis
        lower and upper limits.
    title : str
        Title of the plot.
    path : str
        Path to the saved plot. The path ends with ".png".
    
    """
    # If either `x_values.ndim` or `y_values.ndim` is not
    # equal to 1, `plt.bar` raises an exception.
    plt.bar(x_values,
            y_values,
            width=0.2)
    
    # Matplotlib is forced to display only
    # whole numbers on the x-axis if the
    # x-axis values are integers. Matplotlib
    # is also forced to display only whole
    # numbers on the y-axis if the y-axis
    # values are integers.
    current_axis = plt.gca()
    if numpy.issubdtype(x_values.dtype, numpy.integer):
        current_axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if numpy.issubdtype(y_values.dtype, numpy.integer):
        current_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    current_axis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.ylim(tuple_yaxis_limits)
    plt.xticks(size=22)
    plt.yticks(size=22)
    plt.title(title,
              fontsize=30)
    plt.savefig(path)
    plt.clf()

def plot_graphs(x_values, y_values, x_label, y_label, title, path, legend=None):
    """Overlays several graphs in the same plot and saves the plot.
    
    Parameters
    ----------
    x_values : numpy.ndarray
        1D array.
        x-axis values.
    y_values : numpy.ndarray
        2D array.
        `y_values[i, :]` contains the
        y-axis values of the graph of
        index i.
    x_label : str
        x-axis label.
    y_label : str
        y-axis label.
    title : str
        Title of the plot.
    path : str
        Path to the saved plot. The path ends with ".png".
    legend : list, optional
        `legend[i]` is a string describing the graph of
        index i. The default value is None.
    
    Raises
    ------
    ValueError
        If `x_values.ndim` is not equal to 1.
    ValueError
        If `y_values.ndim` is not equal to 2.
    
    """
    # If `x_values.ndim` and `y_values.ndim` are equal to 2
    # and `x_values.shape[0]` is equal to `y_values.shape[1]`
    # for instance, `plot_graphs` does not crash and saves
    # a wrong plot. That is why `x_values.ndim` and `y_values.ndim`
    # are checked.
    if x_values.ndim != 1:
        raise ValueError('`x_values.ndim` is not equal to 1.')
    if y_values.ndim != 2:
        raise ValueError('`y_values.ndim` is not equal to 2.')
    
    # Matplotlib is forced to display only
    # whole numbers on the x-axis if the
    # x-axis values are integers. Matplotlib
    # is also forced to display only whole
    # numbers on the y-axis if the y-axis
    # values are integers.
    current_axis = plt.figure().gca()
    if numpy.issubdtype(x_values.dtype, numpy.integer):
        current_axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if numpy.issubdtype(y_values.dtype, numpy.integer):
        current_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    
    # For the x-axis or the y-axis, if the range
    # of the absolute values is outside [1.e-4, 1.e4],
    # scientific notation is used.
    plt.ticklabel_format(style='sci',
                         axis='both',
                         scilimits=(-4, 4))
    
    # `plt.plot` returns a list.
    handle = []
    for i in range(y_values.shape[0]):
        handle.append(plt.plot(x_values, y_values[i, :])[0])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(handle, legend)
    plt.savefig(path)
    plt.clf()

def plot_two_rate_distortion_curves(rates_1d_0, psnrs_1d_0, rates_1d_1, psnrs_1d_1, legend, path):
    """Plots two rate-distortion curves and computes Bjontegaard's metric between the two curves.
    
    Parameters
    ----------
    rates_1d_0 : numpy.ndarray
        1D array.
        Rates for the 1st rate-distortion curve. `rates_1d_0[i]`
        is the rate of the point of index i in the 1st rate-distortion
        curve.
    psnrs_1d_0 : numpy.ndarray
        1D array.
        PSNRs for the 1st rate-distortion curve. `psnrs_1d_0[i]`
        is the PSNR of the point of index i in the 1st rate-distortion
        curve.
    rates_1d_1 : numpy.ndarray
        1D array.
        Rates for the 2nd rate-distortion curve. `rates_1d_1[i]`
        is the rate of the point of index i in the 2nd rate-distortion
        curve.
    psnrs_1d_1 : numpy.ndarray
        1D array.
        PSNRs for the 2nd rate-distortion curve. `psnrs_1d_1[i]`
        is the PSNR of the point of index i in the 2nd rate-distortion
        curve.
    legend : list
        Legend of the plot.
    path : str
        Path to the saved plot. The path ends with ".png".
    
    """
    if os.path.isfile(path):
        print('The plot at "{}" already exists.'.format(path))
    else:
        
        # `compute_bjontegaard` checks that `rates_1d_0.ndim` is equal to 1,
        # `rates_1d_1.ndim` is equal to 1, `rates_1d_0.shape` is equal to
        # `psnrs_1d_0.shape` and `rates_1d_1.shape` is equal to `psnrs_1d_1.shape`.
        metric_bjontegaard = compute_bjontegaard(rates_1d_0,
                                                 psnrs_1d_0,
                                                 rates_1d_1,
                                                 psnrs_1d_1)
        handle = [
            plt.plot(rates_1d_0,
                     psnrs_1d_0,
                     color='orange',
                     marker='x',
                     markersize=9.)[0],
            plt.plot(rates_1d_1,
                     psnrs_1d_1,
                     color='red',
                     markerfacecolor='None',
                     marker='o',
                     markeredgecolor='red',
                     markersize=9.)[0]
        ]
        plt.title('Rate-distortion curves | Bjontegaard {}%'.format(round(metric_bjontegaard, 3)),
                  fontsize=20)
        plt.xlabel('Rate (bbp)',
                   fontdict={'size':20})
        plt.ylabel('PSNR (dB)',
                   fontdict={'size':20})
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(handle,
                   legend,
                   loc='lower right',
                   prop={'size': 20},
                   frameon=False)
        plt.savefig(path)
        plt.clf()

def read_image_mode(path, mode):
    """Reads the image if its mode matches the given mode.
    
    Parameters
    ----------
    path : str
        Path to the image to be read.
    mode : str
        Given mode. The two most common modes
        are 'RGB' and 'L'.
    
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.uint8`.
        Image.
    
    Raises
    ------
    ValueError
        If the image mode is not equal to `mode`.
    
    """
    image = PIL.Image.open(path)
    if image.mode != mode:
        raise ValueError('The image mode is {0} whereas the given mode is {1}.'.format(image.mode, mode))
    return numpy.asarray(image)

def rgb_to_ycbcr(rgb_uint8):
    """Converts the RGB image to YCbCr.
    
    The conversion is ITU-T T.871. This means that,
    if the pixels of the RGB image span the range [|0, 255|],
    the pixels of the luminance channel and the pixels of
    the chrominance channels span the range [|0, 255|].
    Note that the OpenCV function `cvtColor` with the
    code `CV_BGR2YCrCb` also implements ITU-T T.871.
    
    Parameters
    ----------
    rgb_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        RGB image.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        YCbCr image.
    
    Raises
    ------
    TypeError
        If `rgb_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `rgb_uint8.ndim` is not equal to 3.
    ValueError
        If `rgb_uint8.shape[2]` is not equal to 3.
    
    """
    if rgb_uint8.dtype != numpy.uint8:
        raise TypeError('`rgb_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If the check below did not exist, `rgb_to_ycbcr` would
    # not crash if `rgb_uint8` is nD, n >= 4.
    if rgb_uint8.ndim != 3:
        raise ValueError('`rgb_uint8.ndim` is not equal to 3.')
    
    # If the check below did not exist, `rgb_to_ycbcr` would
    # not crash if `rgb_uint8.shape[2]` is larger than 4.
    if rgb_uint8.shape[2] != 3:
        raise ValueError('`rgb_uint8.shape[2]` is not equal to 3.')
    rgb_float64 = rgb_uint8.astype(numpy.float64)
    y_float64 = 0.299*rgb_float64[:, :, 0] \
                + 0.587*rgb_float64[:, :, 1] \
                + 0.114*rgb_float64[:, :, 2]
    cb_float64 = 128. \
                 - (0.299/1.772)*rgb_float64[:, :, 0] \
                 - (0.587/1.772)*rgb_float64[:, :, 1] \
                 + (0.886/1.772)*rgb_float64[:, :, 2]
    cr_float64 = 128. \
                 + (0.701/1.402)*rgb_float64[:, :, 0] \
                 - (0.587/1.402)*rgb_float64[:, :, 1] \
                 - (0.114/1.402)*rgb_float64[:, :, 2]
    ycbcr_float64 = numpy.stack((y_float64, cb_float64, cr_float64),
                                axis=2)
    return cast_float_to_uint8(ycbcr_float64)

def save_image(path, array_uint8, coefficient_enlargement=None):
    """Saves the array as an image.
    
    `scipy.misc.imsave` is deprecated in Scipy 1.0.0.
    `scipy.misc.imsave` will be removed in Scipy 1.2.0.
    `save_image` replaces `scipy.misc.imsave`.
    
    Parameters
    ----------
    path : str
        Path to the saved image.
    array_uint8 : numpy.ndarray
        Array with data-type `numpy.uint8`.
        Array to be saved as an image.
    coefficient_enlargement : int, optional
        Coefficient for enlarging the image along
        its first two dimensions before saving it.
        The default value is None, meaning that
        there is no enlargement of the image.
    
    Raises
    ------
    TypeError
        If `array_uint8.dtype` is not equal to `numpy.uint8`.
    
    """
    if array_uint8.dtype != numpy.uint8:
        raise TypeError('`array_uint8.dtype` is not equal to `numpy.uint8`.')
    if coefficient_enlargement is None:
        enlarged_array_uint8 = array_uint8
    else:
        enlarged_array_uint8 = numpy.repeat(numpy.repeat(array_uint8, coefficient_enlargement, axis=0),
                                            coefficient_enlargement,
                                            axis=1)
    image = PIL.Image.fromarray(enlarged_array_uint8)
    image.save(path)

def visualize_channels(channels_uint8, nb_vertically, path):
    """Arranges the same channel of different images in a single image and saves the single image.
    
    Parameters
    ----------
    channels_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. `channels_uint8[i, :, :, :]` is the
        channel of the image of index i. `channels_uint8.shape[3]`
        is equal to 1.
    nb_vertically : int
        Number of channels per column in the single image.
    path : str
        Path to the saved single image. The path ends
        with ".png".
    
    Raises
    ------
    TypeError
        If `channels_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `channels_uint8.shape[0]` is not divisible by
        `nb_vertically`.
    
    """
    if channels_uint8.dtype != numpy.uint8:
        raise TypeError('`channels_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If `channels_uint8.ndim` is not equal to 4, the unpacking
    # below raises a `ValueError` exception.
    # If `channels_uint8.shape[3]` is not equal to 1, `numpy.squeeze`
    # raises a `ValueError` exception.
    (nb_images, height_channel, width_channel, _) = channels_uint8.shape
    
    # If the check below did not exist, the channel of some
    # images may be missing in `image_uint8`.
    if nb_images % nb_vertically != 0:
        raise ValueError('`channels_uint8.shape[0]` is not divisible by `nb_vertically`.')
    
    # `nb_horizontally` has to be an integer.
    nb_horizontally = nb_images//nb_vertically
    image_uint8 = 255*numpy.ones((nb_vertically*(height_channel + 1) + 1, nb_horizontally*(width_channel + 1) + 1), dtype=numpy.uint8)
    for i in range(nb_vertically):
        for j in range(nb_horizontally):
            image_uint8[i*(height_channel + 1) + 1:(i + 1)*(height_channel + 1), j*(width_channel + 1) + 1:(j + 1)*(width_channel + 1)] = \
                numpy.squeeze(channels_uint8[i*nb_horizontally + j, :, :, :],
                              axis=2)
    save_image(path,
               image_uint8)

def visualize_crops(image_uint8, rows_columns_top_left, width_crop, paths):
    """Crops an image several times, repeats the pixels of each crop, and saves the resulting crops.
    
    Parameters
    ----------
    image_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image to be cropped.
    rows_columns_top_left : numpy.ndarray
        2D array whose data-type is smaller than
        `numpy.integer` in type hierarchy.
        `rows_columns_top_left[:, i]` contains
        the row and the column of the image pixel
        at the top-left of the crop of index i.
    width_crop : int
        Width of the crop.
    paths : list
        `paths` is a list of lists of paths. `paths[i][j]`
        is the path to the saved channel of index j of the
        crop of index i. Each path ends with ".png".
    
    Raises
    ------
    ValueError
        If `rows_columns_top_left.shape[0]` is not equal to 2.
    ValueError
        If a crop goes out of the bounds of the image.
    
    """
    # If `image_uint8.ndim` is not equal to 3, the
    # unpacking below raises a `ValueError` exception.
    (height_image, width_image, nb_channels) = image_uint8.shape
    
    # If `rows_columns_top_left.ndim` is not equal to 2
    # the unpacking below raises a `ValueError` exception.
    (height_rows_columns_top_left, nb_crops) = rows_columns_top_left.shape
    if height_rows_columns_top_left != 2:
        raise ValueError('`rows_columns_top_left.shape[0]` is not equal to 2.')
    for i in range(nb_crops):
        row_top_left = rows_columns_top_left[0, i].item()
        column_top_left = rows_columns_top_left[1, i].item()
        if height_image < row_top_left + width_crop or width_image < column_top_left + width_crop:
            raise ValueError('A crop goes out of the bounds of the image.')
        crop_uint8 = image_uint8[row_top_left:row_top_left + width_crop, column_top_left:column_top_left + width_crop, :]
        
        # If `image_uint8.dtype` is not equal to `numpy.uint8`,
        # `save_image` raises a `TypeError` exception.
        for j in range(nb_channels):
            save_image(paths[i][j],
                       crop_uint8[:, :, j],
                       coefficient_enlargement=2)

def visualize_rotated_image(image_uint8, rows_columns_top_left, width_crop, is_rotated, paths):
    """Rotates the image if required, crops the result several times, and saves the rotated image and the crops.
    
    Parameters
    ----------
    image_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image to be cropped.
    rows_columns_top_left : numpy.ndarray
        2D array whose data-type is smaller than
        `numpy.integer` in type hierarchy.
        `rows_columns_top_left[:, i]` contains
        the row and the column of the image pixel
        at the top-left of the crop of index i.
    width_crop : int
        Width of the crop.
    is_rotated : bool
        Is the image rotated?
    paths : list
        `paths` is a list of lists of paths. `paths[0][j]`
        is the path to the saved channel of index j of the
        possibly rotated image. `paths[i + 1][j]` is the path
        to the saved of index j channel of the crop of index i.
        Each path ends with ".png".
    
    """
    if is_rotated:
        image_possible_rotation_uint8 = numpy.rot90(image_uint8,
                                                    k=3)
    else:
        image_possible_rotation_uint8 = image_uint8
    
    # `visualize_crops` checks that `image_possible_rotation_uint8.dtype`
    # is equal to `numpy.uint8` and `image_possible_rotation_uint8.ndim`
    # is equal to 3.
    visualize_crops(image_possible_rotation_uint8,
                    rows_columns_top_left,
                    width_crop,
                    paths[1:])
    for i in range(image_possible_rotation_uint8.shape[2]):
        save_image(paths[0][i],
                   image_possible_rotation_uint8[:, :, i])

def ycbcr_to_rgb(ycbcr_uint8):
    """Converts the YCbCr image to RGB.
    
    The conversion is ITU-T T.871.
    
    Parameters
    ----------
    ycbcr_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        YCbCr image.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        RGB image.
    
    Raises
    ------
    TypeError
        If `ycbcr_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `ycbcr_uint8.ndim` is not equal to 3.
    ValueError
        If `ycbcr_uint8.shape[2]` is not equal to 3.
    
    """
    if ycbcr_uint8.dtype != numpy.uint8:
        raise TypeError('`ycbcr_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If the check below did not exist, `ycbcr_to_rgb` would
    # not crash if `ycbcr_uint8` is nD, n >= 4.
    if ycbcr_uint8.ndim != 3:
        raise ValueError('`ycbcr_uint8.ndim` is not equal to 3.')
    
    # If the check below did not exist, `ycbcr_to_rgb` would
    # not crash if `ycbcr_uint8.shape[2]` is larger than 4.
    if ycbcr_uint8.shape[2] != 3:
        raise ValueError('`ycbcr_uint8.shape[2]` is not equal to 3.')
    ycbcr_float64 = ycbcr_uint8.astype(numpy.float64)
    red_float64 = ycbcr_float64[:, :, 0] \
                  + 1.402*(ycbcr_float64[:, :, 2] - 128.)
    green_float64 = ycbcr_float64[:, :, 0] \
                    - (0.114*1.772*(ycbcr_float64[:, :, 1] - 128.)/0.587) \
                    - (0.299*1.402*(ycbcr_float64[:, :, 2] - 128.)/0.587)
    blue_float64 = ycbcr_float64[:, :, 0] \
                   + 1.772*(ycbcr_float64[:, :, 1] - 128.)
    rgb_float64 = numpy.stack((red_float64, green_float64, blue_float64),
                              axis=2)
    return cast_float_to_uint8(rgb_float64)


