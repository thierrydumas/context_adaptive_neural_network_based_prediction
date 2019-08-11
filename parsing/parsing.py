"""A library containing functions for parsing command-line arguments and options."""

import argparse

# The functions are sorted in alphabetic order.

def float_positive(string):
    """Converts the string into float.
    
    Parameters
    ----------
    string : str
        String to be converted into float.
    
    Returns
    -------
    float
        Float resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted into float.
    ArgumentTypeError
        If the float resulting from the conversion is not positive.
    
    """
    try:
        floating_point = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into float.'.format(string))
    if floating_point < 0.:
        raise argparse.ArgumentTypeError('{} is not positive.'.format(floating_point))
    else:
        return floating_point

def float_strictly_positive(string):
    """Converts the string into float.
    
    Parameters
    ----------
    string : str
        String to be converted into float.
    
    Returns
    -------
    float
        Float resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted into float.
    ArgumentTypeError
        If the float resulting from the conversion is
        not strictly positive.
    
    """
    try:
        floating_point = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into a float.'.format(string))
    if floating_point <= 0.:
        raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(floating_point))
    else:
        return floating_point

def int_positive(string):
    """Converts the string into integer.
    
    Parameters
    ----------
    string : str
        String to be converted into integer.
    
    Returns
    -------
    int
        Integer resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted into integer.
    ArgumentTypeError
        If the integer resulting from the conversion is
        not positive.
    
    """
    try:
        integer = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into an integer.'.format(string))
    if integer < 0.:
        raise argparse.ArgumentTypeError('{} is not positive.'.format(integer))
    else:
        return integer

def int_strictly_positive(string):
    """Converts the string into integer.
    
    Parameters
    ----------
    string : str
        String to be converted into integer.
    
    Returns
    -------
    int
        Integer resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted into integer.
    ArgumentTypeError
        If the integer resulting from the conversion is
        not strictly positive.
    
    """
    try:
        integer = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into an integer.'.format(string))
    if integer <= 0:
        raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(integer))
    else:
        return integer

def tuple_two_positive_integers(string):
    """Converts the string into a tuple of two positive integers.
    
    Parameters
    ----------
    string : str
        If `string` contains two positive integers
        separated by a comma, the output tuple contains
        the two positive integers. If `string` contains
        a single comma, the output tuple is empty.
    
    Returns
    -------
    tuple
        Pair of positive integers.
    
    Raises
    ------
    ArgumentTypeError
        If `string` does not contain two parts
        separated by a comma.
    ArgumentTypeError
        If `string` does not contain two integers
        separated by a comma.
    ArgumentTypeError
        If an integer is not positive.
    
    """
    list_strings = string.split(',')
    if len(list_strings) != 2:
        raise argparse.ArgumentTypeError('"{}" does not contain two parts separated by a comma.'.format(string))
    if not list_strings[0] and not list_strings[1]:
        return ()
    try:
        tuple_two_integers = tuple(map(int, list_strings))
    except ValueError:
        raise argparse.ArgumentTypeError('"{}" does not contain two integers separated by a comma.'.format(string))
    if tuple_two_integers[0] < 0 or tuple_two_integers[1] < 0:
        raise argparse.ArgumentTypeError('An integer is not positive.')
    else:
        return tuple_two_integers


