# -*- coding: utf-8 -*-

import pickle


def save_pkl(obj, filename):
    """
    save object to file using pickle
    
    Don't work well for class istances !
    
    Parameters:
    -----------    
    obj:            object to save
    filename:       string, path of file, use r'.pkl'
    
    
    Examples:
    --------
    save_object( obj , folder+r'\data.pkl')
    
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    """
    load object from file using pickle
    
    Don't work well for class istances !
    
    Parameters:
    -----------    
    filename:       string, path of file, use r'.pkl'
    
    Returns:
    -----------
    obj:            object saved in filename
    
    
    Examples:
    --------
    load_object( obj , folder+r'\data.pkl')
    """
    with open(filename, 'rb') as input:
        return pickle.load(input)
        
