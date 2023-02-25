from importlib import resources 
import ghost

def get_ghost_dir():
    return resources.path(package=ghost, resource="").__enter__()