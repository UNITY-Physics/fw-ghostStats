from importlib import resources 
import os
import ghost

def get_ghost_dir():
    p = os.path.join(resources.path(package=ghost, resource="").__enter__(), '..')
    return os.path.abspath(p)