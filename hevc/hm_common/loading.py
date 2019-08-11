"""A library containing a function for loading a Python object via pickle."""

import pickle

def load_via_pickle(path_to_file):
    """Loads the Python object stored in a file via pickle.
    
    Parameters
    ----------
    path_to_file : str
        Path to the file storing the Python object
        to be loaded via pickle. The path ends with
        ".pkl".
    
    Returns
    -------
    Python object.
    
    """
    with open(path_to_file, 'rb') as file:
        return pickle.load(file)


