"""A library containing functions for extracting archives."""

import os
import six.moves.urllib.request
import tarfile

# The functions are sorted in alphabetic order.

def download_untar_archive(source_url, path_to_directory_extraction, path_to_tar):
    """Downloads the archive if it does not exist at the given path and extracts it.
    
    Parameters
    ----------
    source_url : str
        URL of the archive to be downloaded.
    path_to_directory_extraction : str
        Path to the directory to which the downloaded archive is extracted.
    path_to_tar : str
        Path to the downloaded archive.
    
    Returns
    -------
    bool
        Is the archive downloaded?
    
    """
    if os.path.isfile(path_to_tar):
        is_downloaded = False
    else:
        six.moves.urllib.request.urlretrieve(source_url,
                                             path_to_tar)
        is_downloaded = True
    
    # If the same extraction is run two times in a row,
    # the result of the first extraction is overwritten.
    untar_archive(path_to_directory_extraction,
                  path_to_tar)
    return is_downloaded

def untar_archive(path_to_directory_extraction, path_to_tar):
    """Extracts the archive to the given directory.
    
    Parameters
    ----------
    path_to_directory_extraction : str
        Path to the directory to which the archive is extracted.
    path_to_tar : str
        Path to the archive to be extracted.
        The path ends with ".tar".
    
    """
    with tarfile.open(path_to_tar, 'r') as file:
        file.extractall(path=path_to_directory_extraction)

def untar_ilsvrc2012_training(path_to_directory_extraction, path_to_tar_ilsvrc2012, path_to_synsets):
    """Extracts the file "ILSVRC2012_img_train.tar".
    
    The file "ILSVRC2012_img_train.tar" is first extracted
    to the directory at `path_to_directory_extraction`. This
    puts 1000 files ".tar" into this directory. Then, each
    of these 1000 files is extracted to a subdirectory of the
    directory at `path_to_directory_extraction`.
    
    Parameters
    ----------
    path_to_directory_extraction : str
        Path to the directory to which "ILSVRC2012_img_train.tar"
        is extracted.
    path_to_tar_ilsvrc2012 : str
        Path to the file "ILSVRC2012_img_train.tar".
    path_to_synsets : str
        Path to the file "synsets.txt".
    
    """
    untar_archive(path_to_directory_extraction,
                  path_to_tar_ilsvrc2012)
    with open(path_to_synsets, 'r') as file:
        names_raw = file.readlines()
    
    # Each string in the list `names_raw` ends with "\n".
    names = [name_raw.split('\n', 1)[0] for name_raw in names_raw]
    
    # Each of the 1000 files ".tar" contains the RGB images
    # associated to a class.
    for i, name in enumerate(names):
        path_to_directory_class = os.path.join(path_to_directory_extraction,
                                               name)
        path_to_tar_class = os.path.join(path_to_directory_extraction,
                                         name + '.tar')
        if not os.path.isdir(path_to_directory_class):
            os.mkdir(path_to_directory_class)
            untar_archive(path_to_directory_class,
                          path_to_tar_class)
            print('"{0}" is extracted ({1}).'.format(path_to_tar_class, i + 1))
        else:
            print('"{}" already exists.'.format(path_to_directory_class))
            print('"{}" is not extracted.'.format(path_to_tar_class))


