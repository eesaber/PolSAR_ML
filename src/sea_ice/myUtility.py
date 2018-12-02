import os, sys

def get_path():
    path_root = os.path.dirname(os.path.realpath(__file__))
    path_root = path_root[0:path_root.find('src')]
    path = {}
    path['file'] = path_root+'data/'
    path['log'] = path_root+'log/' # dir for tensorboard
    path['val'] = path_root+'data_val/' # dir of the validation data. 
    path['model'] = path_root+'model/' # dir for saving model.
    path['aug'] = path_root+'data_aug/' # dir of the augmented data.
    path['output'] = path_root+'output/' # dir for saving output data.
    for key, value in path.items():
        if not os.path.exists(value):
            os.makedirs(value)
    return path   