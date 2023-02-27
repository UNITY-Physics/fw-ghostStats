from . import GHOSTDIR
import subprocess as sp
import os

def get_phantom_nii(weighting='T1'):
    if weighting=='T1':
        return os.path.join(GHOSTDIR, '..', 'data', 'T1_phantom.nii.gz')
    elif weighting=='T2':
        return os.path.join(GHOSTDIR, '..', 'data', 'T2_phantom.nii.gz')
    else:
        return ValueError('Not a valid weighting. (Valid: T1, T2)')
    
def download_ref_data():
    files = [{'fname':'T1_phantom.nii.gz',
            'link':'https://www.dropbox.com/s/37ua5rpiv57soz1/T1_phantom.nii.gz?dl=0'},
            {'fname':'T2_phantom.nii.gz',
            'link':'https://www.dropbox.com/s/5v13loxdrkgrlv1/T2_phantom.nii.gz?dl=0'},
            {'fname':'ADC_vials.nii.gz',
            'link':'https://www.dropbox.com/s/yuf0sl9uz1bkqu5/ADC_vials.nii.gz?dl=0'},
            {'fname':'LC_vials.nii.gz',
            'link':'https://www.dropbox.com/s/1c2mjugtyb04xjp/LC_vials.nii.gz?dl=0'},
            {'fname':'T2_vials.nii.gz',
            'link':'https://www.dropbox.com/s/vkkwd02f8dz2nqu/T2_vials.nii.gz?dl=0'},
            {'fname':'T1_vials.nii.gz',
            'link':'https://www.dropbox.com/s/0ai6z3cg94xcvn8/T1_vials.nii.gz?dl=0'}
            ]

    # Check if folder exists
    dl_path = f"{GHOSTDIR}/data"
    if not os.path.exists(dl_path):
        os.mkdir(dl_path)
        print(f"Created folder: {dl_path}")

    for f in files:
        file_path = f"{dl_path}/{f['fname']}"
        if not os.path.exists(file_path):
            dl_link = f['link']
            print('Downloading %s'%(f['fname']))
            cmd = f'wget -q -O {file_path} "{dl_link}"'
            out = sp.call(cmd, shell=True)
            print(f'Done. File saved to {file_path}')

        else:
            print("%s is already downloaded. Skipping"%f['fname'])
    
