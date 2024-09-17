import glob
import json
import os
import shutil

import ants
import pydicom

from .dataio import get_nifti_basename, load_4D_nifti
from .misc import ghost_path
from .phantom import Caliber137
from .web import GHOST_ASSETS, figshare_download

def warp_rois(input, output, seg, weighting, vol, phantom_model, 
              do_syn, xfm_out_name, xfm_aff_in, xfm_syn_in, save_xfm):
        
        # Read input image
        img = load_4D_nifti(input, vol=vol, mag=True)
        
        if output is None:
            output_basename = get_nifti_basename(input)
        else:
            output_basename = output
        
        outname = f'{output_basename}_seg_{seg}.nii.gz'

        print(f"Warping {seg} mask to {input}")
        print(f"Output filename: {outname}")

        # Valid phantoms
        valid_phantoms = ['Caliber137']
        if phantom_model not in valid_phantoms:
            raise ValueError(f"Not a valid phantom. Valid: {valid_phantoms}")

        if phantom_model == 'Caliber137':
            phantom = Caliber137()

        if seg not in phantom.valid_seg:
            raise ValueError(f'{seg} is not a valid segmentation. (Valid: {phantom.valid_seg})')
        
        else:
            if not xfm_aff_in:
                print("No rigid transform provided. Doing registration to template")
                xfm_inv, xfm_fwd = phantom.reg_to_phantom(img, do_syn=do_syn, weighting=weighting, init_z=True)
            else:
                print("Registration transforms provided")
                
                xfm_inv = [xfm_aff_in]
                if xfm_syn_in:
                    xfm_inv.append(xfm_syn_in)

            print("Warping segmentation to input data")
            seg_img = phantom.warp_seg(img, xfm=xfm_inv, seg=seg)

            ants.image_write(seg_img, outname)
            print(f"Saved {outname}")

            if xfm_out_name or save_xfm:
                if not xfm_out_name:
                    xfm_aff_fname = f'{output_basename}_0GenericAffine.mat'
                    xfm_syn_fname = f'{output_basename}_1InverseWarp.nii.gz'
                else:
                    xfm_aff_fname = f'{xfm_out_name}_0GenericAffine.mat'
                    xfm_syn_fname = f'{xfm_out_name}_1InverseWarp.nii.gz'

                shutil.copy(xfm_inv[0], xfm_aff_fname)
                print(f"Saved aff transform to: {xfm_aff_fname}")
                if len(xfm_inv)>1:
                    shutil.copy(xfm_inv[1], xfm_syn_fname)
                    print(f"Saved syn transform to: {xfm_syn_fname}")

def download_ref_data(dl_nnUnet, dl_phantoms, dl_examples, over_write):

    if dl_phantoms:
        avail_phantoms = GHOST_ASSETS['figshare']['PhantomModels']

        for phantom in avail_phantoms:
            download_phantom(phantom, over_write)

    if dl_nnUnet:
        download_nnunet(over_write)

    if dl_examples:
        download_examples(over_write)

def download_examples(over_write):
    """
    Downloads example data from figshare
    """
    gpath = ghost_path()
    print(f"Downloading example data from figshare.")

    fs_info = GHOST_ASSETS['figshare']['ExampleData']['UNITY_QA']
    dl_path = os.path.join(gpath, 'example_data', 'UNITY_QA')
    figshare_download(fs_info['object_id'], fs_info['version'], dl_path, over_write)

    if (not os.path.exists(f"{dl_path}/DICOM")) or over_write:
        print("Extracting DICOM.zip")
        shutil.unpack_archive(f'{dl_path}/DICOM.zip', extract_dir=dl_path, format='zip')
    else:
        print(f"Archive already extracted to: {dl_path}/DICOM")

def download_nnunet(over_write):
    """
    Downloads nnUNet models from figshare
    """
    gpath = ghost_path()
    print(f"Downloading nnUNet models from figshare. Warning! Large files.")

    fs_info = GHOST_ASSETS['figshare']['nnUNet']['Caliber137_fiducials']
    dl_path = os.path.join(gpath, 'nnUNet')
    figshare_download(fs_info['object_id'], fs_info['version'], dl_path, over_write)

def download_phantom(phantom_name, over_write):
    """
    Downloads phantom reference data from figshare
    """
    
    gpath = ghost_path()

    print(f'Downloading reference data for {phantom_name}')

    fs_info = GHOST_ASSETS['figshare']['PhantomModels'][phantom_name]

    # Check if folder exists
    dl_path = os.path.join(gpath, phantom_name)
    if not os.path.exists(dl_path):
        os.makedirs(dl_path)
        print(f"Created folder: {dl_path}")

    figshare_download(fs_info['object_id'], fs_info['version'], dl_path, over_write)

def update_sidecar(args):

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