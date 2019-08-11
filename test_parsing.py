"""A script to test the library "parsing/parsing.py"."""

import argparse

import parsing.parsing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library "parsing/parsing.py".')
    parser.add_argument('float_pos',
                        help='argument to be converted into a positive float',
                        type=parsing.parsing.float_positive)
    parser.add_argument('float_strictly_pos',
                        help='argument to be converted into a strictly positive float',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('int_pos',
                        help='argument to be converted into a positive integer',
                        type=parsing.parsing.int_positive)
    parser.add_argument('int_strictly_pos',
                        help='argument to be converted into a strictly positive integer',
                        type=parsing.parsing.int_strictly_positive)
    parser.add_argument('tuple_two_ints_pos',
                        help='argument to be converted into a tuple of two positive integers',
                        type=parsing.parsing.tuple_two_positive_integers)
    args = parser.parse_args()
    
    print('{} is a positive float.'.format(args.float_pos))
    print('{} is a strictly positive float.'.format(args.float_strictly_pos))
    print('{} is a positive integer.'.format(args.int_pos))
    print('{} is a strictly positive integer.'.format(args.int_strictly_pos))
    print('{} is either an empty tuple or a tuple of two positive integers.'.format(args.tuple_two_ints_pos))


