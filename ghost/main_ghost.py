import argparse
import glob
import json
import os
import re
import sys

import ants
import pydicom
import requests

from .dataio import get_nifti_basename, load_4D_nifti
from .misc import ghost_path
# from .phantom import warp_seg


def main():
    GHOST_parser()

class GHOST_parser(object):

    def __init__(self):
        method_list = [method for method in dir(self) if method.startswith('_') is False]
        
        method_str=''
        for method in method_list:
            dstr = eval(f'self.{method}.__doc__')
            method_str += "\t{:25s}{:25s}\n".format(method, dstr)

        parser = argparse.ArgumentParser(description='GHOST: A framework for phantom analysis in the UNITY project',
                                         usage=f'''ghost <command> [<args>]

    Available commands are
{method_str}

    ''')
        
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        # Check if the object (the class) has a function with the given command name
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # Call the method
        getattr(self, args.command)()

    def warp_rois(self):
        """Warp ROIS"""
        parser = argparse.ArgumentParser(description='Warp ROIs to target image',
                                         usage='ghost warp_rois <input> [<args>]]')
        
        parser.add_argument('input', type=str, help='Input image')
        parser.add_argument('-w', '--weighting', type=str, default='T1', help='Phantom weighting (T1 or T2)')
        parser.add_argument('-s', '--seg', action='append', type=str, help='Segmentation (T1, T2, ADC)')
        parser.add_argument('-o', '--out', type=str, default=None, help='Output basename (default is input basename)')
        parser.add_argument('--vol', type=int, default=None, help='Volume to use (default is last volume)')
        
        args = parser.parse_args(sys.argv[2:])
        main_warp_rois(args)

    def setup(self):
        """Download data"""
        parser = argparse.ArgumentParser(description='Setup repo and download data',
                                         usage='ghost setup')
        args = parser.parse_args(sys.argv[2:])
        main_setup(args)

    def update_sidecar(self):
        """Update json sidecar info from dicom tags"""
        parser = argparse.ArgumentParser(description='Update sidecar json files with info from dicom files')
        parser.add_argument('dicom_dir', help='Directory containing dicom files')
        parser.add_argument('json_dir', help='Directory containing json files')
        parser.add_argument('-m', '--matches', help='File containing matches between dicom tags and json fields', required=True)
        parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
        
        args = parser.parse_args(sys.argv[2:])
        main_update_sidecar(args)


def main_warp_rois(args):
        # Read input image
        img = load_4D_nifti(args.input, vol=args.vol, mag=True)
        
        # Check segmentation options
        valid_segs = ['T1', 'T2', 'ADC', 'BG']
        if args.out is None:
            output_basename = get_nifti_basename(args.input)
        else:
            output_basename = args.out
        for s in args.seg:
            if s not in valid_segs:
                raise ValueError(f'Not a valid segmentation. (Valid: {valid_segs})')
            else:
                seg = warp_seg(img, weighting=args.weighting, seg=s)
                outname = f'{output_basename}_mask{s}.nii.gz'
                ants.image_write(seg, outname)
                print(f"Saved {outname}")

def main_setup(args):
    download_ref_data()

def download_ref_data():
    """
    Downloads reference data for the phantom from Dropbox
    """
    files = [{'fname':'phantom_T1w.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/z0tpp56psmpo86fcahn7u/phantom_T1w.nii.gz?rlkey=p3ii0o2ya4lwy3phq9p0ftjkc&st=7kz19xv8&dl=1'},
            {'fname':'phantom_T2w.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/o2l2bh2te10vzs1z56izd/phantom_T2w.nii.gz?rlkey=ei6ephfc72y8u4ep0w1bvq1zj&st=evls4fwm&dl=1'},
            {'fname':'seg_ADC.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/pmf0qc9tk3t613ss9xqsu/seg_ADC.nii.gz?rlkey=evb2kb63zpv66yhth12d64lie&st=ucq6jgpn&dl=1'},
            {'fname':'seg_BG.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/brz47of6y4snunm5omeck/seg_BG.nii.gz?rlkey=6eszykr227ghll26jbs6pf201&st=2qid8jyq&dl=1'},
            {'fname':'seg_fiducials.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/db5cx88lv6z2rw46jsw2u/seg_fiducials.nii.gz?rlkey=echqhdav6jw3acpjp676grn0v&st=4ompzy84&dl=1'},
            {'fname':'seg_LC.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/hr2m5mqzzzcqmi5r2hwhy/seg_LC.nii.gz?rlkey=tojw4b1vdurheox7u8iqw3pyo&st=i2etmxrb&dl=1'},
            {'fname':'seg_phantomMask',
            'link':'https://www.dropbox.com/scl/fi/tin0fcvt1mni3tbll5a1z/seg_phantomMask.nii.gz?rlkey=zbot4ggmca4hjmtpc1na8e6q4&st=rplsy4jc&dl=1'},
            {'fname':'seg_SNR.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/b7516jr8nb6h1th19f1ne/seg_SNR.nii.gz?rlkey=q01be17mes45yt7ntqmmdt3sc&st=h0gmwif6&dl=1'},
            {'fname':'seg_T1mimics.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/ozq99zdbsoooyv5z3ouvf/seg_T1mimics.nii.gz?rlkey=4bkwn0o5gzwnc7wgqutcs7r9d&st=svycvl5h&dl=1'},
            {'fname':'seg_T2mimics.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/209of6hxdgi808m4qwl08/seg_T2mimics.nii.gz?rlkey=at6rjrzua9aluyh37rozm513n&st=l98hl6jp&dl=1'},
            {'fname':'seg_wedges.nii.gz',
            'link':'https://www.dropbox.com/scl/fi/4gj6kc0ufyq0lmqoahiy8/seg_wedges.nii.gz?rlkey=xo8drjz6xmb9shlumbu459onp&st=t0hfc6wv&dl=1'},
            {'fname':'spec.json',
             'link':'https://www.dropbox.com/scl/fi/7lyawts3x5c3htvsfc0se/spec.json?rlkey=9v98kmn7hgod59hphomltyyqw&st=r72okmox&dl=0'}
            ]

    # Check if folder exists
    dl_path = os.path.join(ghost_path(), 'data', 'Caliber137')
    if not os.path.exists(dl_path):
        os.makedirs(dl_path)
        print(f"Created folder: {dl_path}")

    for f in files:
        file_path = f"{dl_path}/{f['fname']}"
        if not os.path.exists(file_path):
            dl_link = f['link']
            print('Downloading %s from %s'%(f['fname'], f['link']))
            myfile = requests.get(dl_link)
            fwrite = open(file_path, 'wb').write(myfile.content)
            print(f'Done. File saved to {file_path}')

        else:
            print("%s is already downloaded. Skipping"%f['fname'])


def main_update_sidecar(args):

    def cast(x, dtype):
        if dtype == 'str':
            return str(x)
        elif dtype == 'int':
            return int(x)
        elif dtype == 'float':
            return float(x)

    def parse_line(line):
        line = [x.strip() for x in line.split(',')]
        for i in range(len(line)):
            if line[i] == 'None':
                line[i] = None
        return line

    def get_dcm_value(dcm, tag, dtype, reg_ex, name):
        
        tag_list = tag.split('/')
        n_tags = len(tag_list)
        val = dcm
        tag_idx = 0
        for tag in tag_list:
            val = val[tag]
            if tag_idx < n_tags-1:
                val = val[0]
            tag_idx += 1
            
        val = val.value
        if reg_ex is not None:
            val = re.findall(rf'{reg_ex}', val)[0]
        
        if name is not None:
            tag_name = name
        else:
            tag_name = dcm[tag].name
        
        out = {tag_name: cast(val, dtype)}
        return out

    def vprint(s, v):
        if v: print(s)

    vp = args.verbose

    json_dir = args.json_dir
    dicom_dir = args.dicom_dir
    json_files = glob.glob(f'{json_dir}/**/*.json', recursive=True)
    dicom_files = glob.glob(f'{dicom_dir}/**/*.dcm', recursive=True)

    vprint(f"Found {len(json_files)} json files in {json_dir}", vp)
    vprint(f"Found {len(dicom_files)} dicom files in {dicom_dir}", vp)

    if len(json_files) == 0:
        raise FileNotFoundError('No json files found in {}'.format(json_dir))
    if len(dicom_files) == 0:
        raise FileNotFoundError('No dicom files found in {}'.format(dicom_dir))

    vprint('Matching nifti and dicom files on series number', vp)
    json_numbers = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            series_number = json.load(f)['SeriesNumber']
            json_numbers[series_number] = json_file

    dicom_numbers = {}
    for dicom_file in dicom_files:
        dcm = pydicom.read_file(dicom_file)
        dicom_numbers[int(dcm['0x00200011'].value)] = dicom_file

    file_pairs = []
    for series_number in json_numbers.keys():
        if series_number in dicom_numbers:
            file_pairs.append((json_numbers[series_number], dicom_numbers[series_number]))
        else:
            vprint(f"Could not find dicom file for series number {series_number}", vp)

    parser = {}
    vprint("Applying the following matches", vp)
    vprint(f"DicomTag\t\tdtype\t\tregex\t\tname", vp)
    with open(args.matches, 'r') as f:
        f.readline() # Remove header
        for line in f:
            line = parse_line(line)
            parser[line[0]] = {'dtype':line[1], 'reg_ex':line[2], 'name':line[3]}
            vprint(f"{line[0]}\t\t{line[1]}\t\t{line[2]}\t\t{line[3]}", vp)

    vprint("Updating json sidecar files", vp)
    for i in range(len(file_pairs)):
        jfile,dfile = file_pairs[i]
        vprint(f"{dfile} -> {jfile}", vp)

        with open(jfile, 'r+') as f:
            jdata = json.load(f)
            dcm = pydicom.read_file(dfile)

            for p in parser.keys():
                D = get_dcm_value(dcm, p, **parser[p])
                jdata.update(D)
        with open(jfile, 'w') as f:
            json.dump(jdata, f, indent=4)


if __name__ == '__main__':
    main()
