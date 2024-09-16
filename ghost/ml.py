import json
import os
import shutil
import subprocess as sp
import tempfile


def run_prediction(input, output, scan_plane, config, device='cuda', keep=False):

    scan_plane = scan_plane.lower()
    valid_scan_planes = ['axi', 'sag', 'cor']
    if scan_plane not in valid_scan_planes:
        raise ValueError(f'Scan plan is not valid. Should be {valid_scan_planes}')

    if type(input) == str:
        input = [input]
        output = [output]

    if len(input) != len(output):
        raise ValueError('Input and output must be the same length')
    
    # 1. Create temporary directories
    print(f"Running inference on {len(input)} images")

    tmpdir = tempfile.mkdtemp(dir='.', prefix='tempdir_nnUNet_inference_')
    os.makedirs(tmpdir+'/input')
    os.makedirs(tmpdir+'/predicted')
    os.makedirs(tmpdir+'/final')
    print(f"Creating temporary directory for processin {tmpdir}")

    channel = '0000'
    data_prefix = 'UNITY'
    data_idx = [f'{int(x):04d}' for x in range(1,len(input)+1)]
    
    # Temporary copy files to folder with nnUNet name standard
    for i in range(len(input)):
        shutil.copy(input[i], os.path.join(tmpdir, 'input', f'{data_prefix}_{data_idx[i]}_{channel}.nii.gz'))

    # 2. Get inference parameters
    with open(config, 'r') as f:
        jdata = json.load(f)[scan_plane]

    nnUNet_results = os.getenv("nnUNet_results")

    # 3. Run prediction
    print("Starting prediction")
    
    cmd_predict = f"nnUNetv2_predict --verbose -device {device} -d {jdata['dataset_id']} -i {tmpdir+'/input'} -o {tmpdir+'/predicted'} -f {jdata['folds']} -tr {jdata['trainer']} -c {jdata['config']} -p {jdata['plan']} --disable_progress_bar"
    
    sp.call(cmd_predict, shell=True)
    
    # 4. Run post processing
    print("Starting post-processing")
    
    cmd_process = f"nnUNetv2_apply_postprocessing -i {tmpdir+'/predicted'} -o {tmpdir+'/final'} -pp_pkl_file {nnUNet_results}/{jdata['pp_pkl_file']} -np 8 -plans_json {nnUNet_results}/{jdata['plans_json']}"
    
    sp.call(cmd_process, shell=True)

    # Save data
    for idx,f in zip(data_idx, output):
        try:
            shutil.copy(f'{tmpdir}/final/{data_prefix}_{idx}.nii.gz', f)
            print(f)
        except:
            print(f'Could not copy to dest {f}')
    if not keep:
        print("Cleaning up temporary directory")
        shutil.rmtree(tmpdir)