from pathlib import Path

def ghost_path():
    gpath = Path.home().joinpath('ghost_data')
    
    try:
        gpath.mkdir()
    except:
        pass

    return str(gpath)
    
