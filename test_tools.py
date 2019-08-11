"""A script to test the library "tools/tools.py"."""

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

import tools.tools as tls


class TesterTools(object):
    """Class for testing the library "tools/tools.py"."""
    
    def test_cast_float_to_uint8(self):
        """Tests the function `cast_float_to_uint8`.
        
        The test is successful if the data-type cast
        sets the array elements that are larger than 255.0
        to 255 and the array elements that are smaller
        than 0.0 to 0.
        
        """
        array_0_float = numpy.array([[15.431, -0.001, 0.], [235.678, 257.18, 821.12]], dtype=numpy.float32)
        array_0_uint8 = tls.cast_float_to_uint8(array_0_float)
        array_1_float = numpy.array([0., -0.1, 307.1, 1.2], dtype=numpy.float16)
        array_1_uint8 = tls.cast_float_to_uint8(array_1_float)
        array_2_float = numpy.array([2345., -111.2], dtype=numpy.float64)
        array_2_uint8 = tls.cast_float_to_uint8(array_2_float)
        print('1st array before the data-type cast:')
        print(array_0_float)
        print('1st array after the data-type cast:')
        print(array_0_uint8)
        print('2nd array before the data-type cast:')
        print(array_1_float)
        print('2nd array after the data-type cast:')
        print(array_1_uint8)
        print('3rd array before the data-type cast:')
        print(array_2_float)
        print('3rd array after the data-type cast:')
        print(array_2_uint8)
    
    def test_ceil_float(self):
        """Tests the function `ceil_float`.
        
        The test is successful if, for each float, the ceiling
        is correct.
        
        """
        input_float_0 = 3.325798453
        nb_digits_0 = 7
        input_float_1 = -1.219286352
        nb_digits_1 = 8
        
        output_0 = tls.ceil_float(input_float_0,
                                  nb_digits_0)
        output_1 = tls.ceil_float(input_float_1,
                                  nb_digits_1)
        print('1st float: {}'.format(input_float_0))
        print('1st float ceiled to {0} digits after the decimal point: {1}'.format(nb_digits_0, output_0))
        print('2nd float: {}'.format(input_float_1))
        print('2nd float ceiled to {0} digits after the decimal point: {1}'.format(nb_digits_1, output_1))
    
    def test_check_remove_zero_slices_in_array_3d(self):
        """Tests the function `check_remove_zero_slices_in_array_3d`.
        
        The test is successful if the zero slice of index 1
        is removed.
        
        """
        array_3d = numpy.zeros((3, 4, 2))
        array_3d[0, 1, 0] = 1.
        are_zero_columns_bools = numpy.array([False, True])
        array_3d_without_zero_slices = tls.check_remove_zero_slices_in_array_3d(array_3d,
                                                                                are_zero_columns_bools)
        print('3D array before removing its zero slices:')
        print(array_3d)
        print('3D array after removing its zero slices:')
        print(array_3d_without_zero_slices)
    
    def test_clean_sort_list_strings(self):
        """Tests the function `clean_sort_list_strings`.
        
        The test is successful if the cleaned and sorted list of
        strings is ['file_1.png', 'file_3.pgm'].
        
        """
        list_strings = ['file_0.jpg', 'file_1.png', 'file_2.pkl', 'file_3.pgm']
        extension = ('.png', '.pgm')
        
        list_strings_cleaned_sorted = tls.clean_sort_list_strings(list_strings,
                                                                  extension)
        print('List of strings:')
        print(list_strings)
        print('Extensions: {}'.format(extension))
        print('Cleaned and sorted list of strings:')
        print(list_strings_cleaned_sorted)
    
    def test_collect_integer_between_tags_in_each_filename(self):
        """Tests the function `collect_integer_between_tags_in_each_filename`.
        
        The test is successful if the list of integers
        returned by `collect_integer_between_tags_in_each_filename`
        is [0, 31].
        
        """
        path_to_directory = 'tools/pseudo_data/collect_integer_between_tags_in_each_filename/'
        extensions = ('.pkl', '.fake')
        tag_0 = 'storing_'
        tag_1 = '.pkl'
        
        list_integers = tls.collect_integer_between_tags_in_each_filename(path_to_directory,
                                                                          extensions,
                                                                          tag_0,
                                                                          tag_1)
        print('List of integers returned by `collect_integer_between_tags_in_each_filename`:')
        print(list_integers)
    
    def test_collect_paths_to_files_in_subdirectories(self):
        """Tests the function `collect_paths_to_files_in_subdirectories`.
        
        The test is successful if the list
        contains the paths to all the JPEG
        images in the subdirectories of the
        directory "tools/pseudo_data/collect_paths_to_files_in_subdirectories/".
        
        """
        path_to_directory = 'tools/pseudo_data/collect_paths_to_files_in_subdirectories/'
        extension = '.jpg'
        
        paths_to_files = tls.collect_paths_to_files_in_subdirectories(path_to_directory,
                                                                      extension)
        print('List storing the paths to all the JPEG images in the subdirectories of "{}":'.format(path_to_directory))
        print(paths_to_files)
    
    def test_collect_paths_to_subdirectories(self):
        """Tests the function `collect_paths_to_subdirectories`.
        
        The test is successful if the list contains
        the paths to all the subdirectories of the
        directory "tools/pseudo_data/collect_paths_to_subdirectories/".
        
        """
        path_to_directory = 'tools/pseudo_data/collect_paths_to_subdirectories/'
        
        paths_to_subdirectories = tls.collect_paths_to_subdirectories(path_to_directory)
        print('List storing the paths to the subdirectories of "{}":'.format(path_to_directory))
        print(paths_to_subdirectories)
    
    def test_compute_bjontegaard(self):
        """Tests the function `compute_bjontegaard`.
        
        A plot is saved at
        "tools/pseudo_visualization/compute_bjontegaard.png".
        The test is successful if Bjontegaard's metric
        is positive when the 1st rate-distortion curve
        generates bitrate savings with respect to the
        2nd rate-distortion curve.
        
        """
        rates_0 = numpy.linspace(0.15, 2.05, num=191)
        rates_1 = numpy.linspace(0.1, 1.7, num=321)
        psnrs_0 = 40.*numpy.sqrt(rates_0)
        psnrs_1 = 20.*numpy.sqrt(rates_1) + 10.
        
        # If the Bjontegaard metric is positive, the
        # 1st rate-distortion saves bitrate with respect
        # to the 2nd rate-distortion curve.
        metric_bjontegaard = tls.compute_bjontegaard(rates_0,
                                                     psnrs_0,
                                                     rates_1,
                                                     psnrs_1)
        handle = [
            plt.plot(rates_0, psnrs_0)[0],
            plt.plot(rates_1, psnrs_1)[0]
        ]
        plt.title('Two RD curves - Bjontegaard metric: {}%'.format(metric_bjontegaard))
        plt.legend(handle, ['1st rate-distortion curve', '2nd rate-distortion curve'])
        plt.xlabel('rate (bbp)')
        plt.ylabel('PSNR (dB)')
        plt.savefig('tools/pseudo_visualization/compute_bjontegaard.png')
        plt.clf()
    
    def test_compute_psnr(self):
        """Tests the function `compute_psnr`.
        
        The test is successful if the PSNR computed by
        hand is equal to the PSNR computed by `compute_psnr`.
        
        """
        array_0_uint8 = 12*numpy.ones((2, 2), dtype=numpy.uint8)
        array_0_uint8[0, 0] = 5
        array_1_uint8 = 15*numpy.ones((2, 2), dtype=numpy.uint8)
        psnr = tls.compute_psnr(array_0_uint8,
                                array_1_uint8)
        print('PSNR computed by hand: {}'.format(33.1133661756))
        print('PNSR computed by `compute_psnr`: {}'.format(psnr))
    
    def test_divide_ints_check_divisible(self):
        """Tests the function `divide_ints_check_divisible`.
        
        The test is successful if a `ValueError`
        exception is raised when the numerator
        is not divisible by the denominator.
        
        """
        numerator = 400
        denominator = -20
        
        result = tls.divide_ints_check_divisible(numerator,
                                                 denominator)
        print('Numerator: {}'.format(numerator))
        print('Denominator: {}'.format(denominator))
        print('Result of the division: {}'.format(result))
    
    def test_float_to_str(self):
        """Tests the function `float_to_str`.
        
        The test is successful if, for each
        float to be converted, "." is replaced
        by "dot" if the float is not a whole
        number and "-" is replaced by "minus".
        
        """
        float_0 = 2.3
        print('1st float to be converted: {}'.format(float_0))
        print('1st string: {}'.format(tls.float_to_str(float_0)))
        float_1 = -0.01
        print('2nd float to be converted: {}'.format(float_1))
        print('2nd string: {}'.format(tls.float_to_str(float_1)))
        float_2 = 3.
        print('3rd float to be converted: {}'.format(float_2))
        print('3rd string: {}'.format(tls.float_to_str(float_2)))
        float_3 = 0.
        print('4th float to be converted: {}'.format(float_3))
        print('4th string: {}'.format(tls.float_to_str(float_3)))
        float_4 = -4.
        print('5th float to be converted: {}'.format(float_4))
        print('5th string: {}'.format(tls.float_to_str(float_4)))
    
    def test_histogram(self):
        """Tests the function `histogram`.
        
        A histogram is saved at
        "tools/pseudo_visualization/histogram.png".
        The test is successful if the selected
        number of bins (60) gives a good histogram
        of 2000 data points.
        
        """
        data = numpy.random.normal(loc=0.,
                                   scale=1.,
                                   size=2000)
        tls.histogram(data,
                      'Standard normal distribution',
                      'tools/pseudo_visualization/histogram.png')
    
    def test_plot_bars_yaxis_limits(self):
        """Tests the function `plot_bars_yaxis_limits`.
        
        A bar plot is saved at "tools/pseudo_visualization/plot_bars_yaxis_limits.png".
        The test is successful if, in this bar plot, the
        y-axis ticks belong to [-1.0, 0.3].
        
        """
        x_values = numpy.array([0, 1, 2, 3], dtype=numpy.int32)
        y_values = numpy.array([-0.3, 0.01, -0.4, 0.27])
        tuple_yaxis_limits = (-1., 0.3)
        
        tls.plot_bars_yaxis_limits(x_values,
                                   y_values,
                                   tuple_yaxis_limits,
                                   'Test bars',
                                   'tools/pseudo_visualization/plot_bars_yaxis_limits.png')
    
    def test_plot_graphs(self):
        """Tests the function `plot_graphs`.
        
        A plot is saved at "tools/pseudo_visualization/plot_graphs.png".
        The test is successful if the plot
        does not use scientific notation.
        
        """
        x_values = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        y_values = numpy.array(
            [[1.e-4, -1.e-4, 1.e-4, -1.e-4, 1.e-4, -1.e-4],
             [1.e-3, -1.e-3, 1.e-3, -1.e-3, 1.e-3, 1.e-3]]
        )
        tls.plot_graphs(x_values,
                        y_values,
                        'x-values',
                        'y-values',
                        'Tests curves',
                        'tools/pseudo_visualization/plot_graphs.png',
                        legend=['1st test', '2nd test'])
    
    def test_plot_two_rate_distortion_curves(self):
        """Tests the function `plot_two_rate_distortion_curves`.
        
        A plot is saved at
        "tools/pseudo_visualization/plot_two_rate_distortion_curves.png".
        The test is successful if the 1st rate-distortion
        curve is above the 2nd one in this plot.
        
        """
        rates_1d_0 = numpy.linspace(0.2, 2., num=10)
        rates_1d_1 = numpy.linspace(0.1, 2., num=20)
        psnrs_1d_0 = 40.*numpy.sqrt(rates_1d_0)
        psnrs_1d_1 = 20.*numpy.sqrt(rates_1d_1) + 10.
        legend = [
            '1st rate-distortion curve',
            '2nd rate-distortion curve'
        ]
        
        tls.plot_two_rate_distortion_curves(rates_1d_0,
                                            psnrs_1d_0,
                                            rates_1d_1,
                                            psnrs_1d_1,
                                            legend,
                                            'tools/pseudo_visualization/plot_two_rate_distortion_curves.png')
    
    def test_read_image_mode(self):
        """Tests the function `read_image_mode`.
        
        The test is successful if the file "tools/pseudo_data/rgb_web.jpg"
        is read normally. However, the reading of "tools/pseudo_data/cmyk_snake.jpg",
        the reading of "tools/pseudo_data/cmyk_mushroom.jpg" and the
        reading of "tools/pseudo_data/cmyk_hourglass.jpg" each raises
        a `ValueError` exception.
        
        """
        path_to_rgb = 'tools/pseudo_data/rgb_web.jpg'
        paths_to_cmyks = [
            'tools/pseudo_data/cmyk_snake.jpg',
            'tools/pseudo_data/cmyk_mushroom.jpg',
            'tools/pseudo_data/cmyk_hourglass.jpg'
        ]
        
        rgb_uint8 = tls.read_image_mode(path_to_rgb,
                                        'RGB')
        print('The reading of "{0}" yields a Numpy array with shape {1} and data-type {2}.'.format(path_to_rgb, rgb_uint8.shape, rgb_uint8.dtype))
        for path_to_cmyk in paths_to_cmyks:
            try:
                cmyk_uint8 = tls.read_image_mode(path_to_cmyk,
                                                 'RGB')
            except ValueError as err:
                print('The reading of "{}" raises a `ValueError` exception.'.format(path_to_cmyk))
                print(err)
    
    def test_rgb_to_ycbcr(self):
        """Tests the function `rgb_to_ycbcr`.
        
        The test is successful if, for each channel
        Y, Cb and Cr, the channel computed by hand
        is equal to the channel computed by `rgb_to_ycbcr`.
        
        """
        red_uint8 = numpy.array([[1, 214, 23], [45, 43, 0]], dtype=numpy.uint8)
        green_uint8 = numpy.array([[255, 255, 23], [0, 13, 0]], dtype=numpy.uint8)
        blue_uint8 = numpy.array([[100, 255, 0], [0, 0, 0]], dtype=numpy.uint8)
        rgb_uint8 = numpy.stack((red_uint8, green_uint8, blue_uint8),
                                axis=2)
        ycbcr_uint8 = tls.rgb_to_ycbcr(rgb_uint8)
        print('Red channel:')
        print(red_uint8)
        print('Green channel:')
        print(green_uint8)
        print('Blue channel:')
        print(blue_uint8)
        print('Luminance computed by the function:')
        print(ycbcr_uint8[:, :, 0])
        print('Luminance computed by hand:')
        print(numpy.array([[161, 243, 20], [13, 20, 0]], dtype=numpy.uint8))
        print('Blue chrominance computed by the function:')
        print(ycbcr_uint8[:, :, 1])
        print('Blue chrominance computed by hand:')
        print(numpy.array([[93, 135, 117], [120, 116, 128]], dtype=numpy.uint8))
        print('Red chrominance computed by the function:')
        print(ycbcr_uint8[:, :, 2])
        print('Red chrominance computed by hand:')
        print(numpy.array([[14, 108, 130], [150, 144, 128]], dtype=numpy.uint8))
    
    def test_save_image(self):
        """Tests the function `save_image`.
        
        An image is saved at "tools/pseudo_visualization/save_image.png".
        The test is successful if this image is
        identical to "tools/pseudo_data/rgb_web.png".
        
        """
        rgb_uint8 = tls.read_image_mode('tools/pseudo_data/rgb_web.jpg',
                                        'RGB')
        tls.save_image('tools/pseudo_visualization/save_image.png',
                       rgb_uint8)
    
    def test_visualize_channels(self):
        """Tests the function `visualize_channels`.
        
        An image is saved at
        "tools/pseudo_visualization/visualize_channels.png".
        The test is successful if this image
        contains 4 luminance images which are
        arranged in a 4x4 grid.
        
        """
        channels_uint8 = numpy.load('tools/pseudo_data/visualize_channels/luminances_uint8.npy')
        tls.visualize_channels(channels_uint8,
                               2,
                               'tools/pseudo_visualization/visualize_channels.png')
    
    def test_visualize_crops(self):
        """Tests the function `visualize_crops`.
        
        A 1st group of three images is saved at
        "tools/pseudo_visualization/visualize_crops/crop_i.png",
        i = 0 ... 2.
        A 2nd group of three images is saved at
        "tools/pseudo_visualization/visualize_crops/crop_i.png",
        i = 0 ... 2.
        The test is successful if the 1st group of
        three images corresponds to the RGB channels
        of an enlarged crop of the bottom-left of the
        RGB image at "tools/pseudo_data/rgb_web.jpg".
        Besides, the 2nd group of three images must
        correspond to the RGB channels of an enlarged
        crop of the top-right of this image.
        
        """
        width_crop = 80
        paths = [
            ['tools/pseudo_visualization/visualize_crops/crop_0_{}.png'.format(i) for i in range(3)],
            ['tools/pseudo_visualization/visualize_crops/crop_1_{}.png'.format(i) for i in range(3)]
        ]
        
        image_uint8 = tls.read_image_mode('tools/pseudo_data/rgb_web.jpg',
                                          'RGB')
        rows_columns_top_left = numpy.array([[image_uint8.shape[0] - width_crop, 0], [0, image_uint8.shape[1] - width_crop]],
                                            dtype=numpy.int32)
        tls.visualize_crops(image_uint8,
                            rows_columns_top_left,
                            width_crop,
                            paths)
    
    def test_visualize_rotated_image(self):
        """Tests the function `visualize_rotated_image`.
        
        A 1st group of three images is saved at
        "tools/pseudo_visualization/visualize_rotated_image/rotated_channel_i.png",
        i = 0 ... 2.
        A 2nd group of three images is saved at
        "tools/pseudo_visualization/visualize_rotated_image/crop_0_i.png",
        i = 0 ... 2.
        A 3rd group of three images is saved at
        "tools/pseudo_visualization/visualize_rotated_image/crop_1_i.png",
        i = 0 ... 2.
        The test is successful if the 2nd and the 3rd
        groups each corresponds to the RGB channels of
        an enlarged crop of the RGB image at "tools/pseudo_data/rgb_web.jpg".
        
        """
        rows_columns_top_left = numpy.array([[0, 0], [100, 200]], dtype=numpy.int32)
        width_crop = 60
        paths_to_saved_rotated_image_crops = [
            ['tools/pseudo_visualization/visualize_rotated_image/rotated_channel_{}.png'.format(i) for i in range(3)],
            ['tools/pseudo_visualization/visualize_rotated_image/crop_0_{}.png'.format(i) for i in range(3)],
            ['tools/pseudo_visualization/visualize_rotated_image/crop_1_{}.png'.format(i) for i in range(3)]
        ]
        
        image_uint8 = tls.read_image_mode('tools/pseudo_data/rgb_web.jpg',
                                          'RGB')
        tls.visualize_rotated_image(image_uint8,
                                    rows_columns_top_left,
                                    width_crop,
                                    True,
                                    paths_to_saved_rotated_image_crops)
    
    def test_ycbcr_to_rgb(self):
        """Tests the function `ycbcr_to_rgb`.
        
        The test is successful if, for each
        channel red, green and blue, the channel
        computed by hand is equal to the channel
        computed by the function.
        
        """
        y_uint8 = numpy.array([[161, 243, 20], [13, 20, 0]], dtype=numpy.uint8)
        cb_uint8 = numpy.array([[93, 135, 117], [120, 116, 128]], dtype=numpy.uint8)
        cr_uint8 = numpy.array([[14, 108, 130], [150, 144, 128]], dtype=numpy.uint8)
        ycbcr_uint8 = numpy.stack((y_uint8, cb_uint8, cr_uint8),
                                  axis=2)
        rgb_uint8 = tls.ycbcr_to_rgb(ycbcr_uint8)
        print('Luminance:')
        print(y_uint8)
        print('Blue chrominance:')
        print(cb_uint8)
        print('Red chrominance:')
        print(cr_uint8)
        print('Red channel computed by the function:')
        print(rgb_uint8[:, :, 0])
        print('Red channel computed by hand:')
        print(numpy.array([[1, 215, 23], [44, 42, 0]], dtype=numpy.uint8))
        print('Green channel computed by the function:')
        print(rgb_uint8[:, :, 1])
        print('Green channel computed by hand:')
        print(numpy.array([[254, 255, 22], [0, 13, 0]], dtype=numpy.uint8))
        print('Blue channel computed by the function:')
        print(rgb_uint8[:, :, 2])
        print('Blue channel computed by hand:')
        print(numpy.array([[99, 255, 1], [0, 0, 0]], dtype=numpy.uint8))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library "tools/tools.py".')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterTools()
    getattr(tester, 'test_' + args.name)()


