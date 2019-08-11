"""A script to extract the file "ILSVRC2012_img_train.tar"."""

import argparse

import sets.untar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts the file "ILSVRC2012_img_train.tar".')
    parser.add_argument('path_to_directory_extraction',
                        help='path to the directory to which "ILSVRC2012_img_train.tar" is extracted')
    parser.add_argument('path_to_tar_ilsvrc2012',
                        help='path to the file "ILSVRC2012_img_train.tar"')
    parser.add_argument('path_to_synsets',
                        help='path to the file "synsets.txt"')
    args = parser.parse_args()
    
    sets.untar.untar_ilsvrc2012_training(args.path_to_directory_extraction,
                                         args.path_to_tar_ilsvrc2012,
                                         args.path_to_synsets)


