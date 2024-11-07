import os
import shlex
import shutil
import subprocess as sp
import tempfile

from .misc import ghost_path


def update_env():
    env = os.environ
    my_nnUNet_base = ghost_path() + '/nnUNet'
    for p in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        env[p] = os.path.join(my_nnUNet_base, p)
    
    return env

def run_prediction(input, output, scan_plane, device='cuda', keep=False, verbose=False):

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
    jdata = get_model_config(scan_plane)

    env = update_env()
    nnUNet_results = env["nnUNet_results"]

    # 3. Run prediction
    print("Starting prediction")
    vstring = '--verbose' if verbose else ''
    
    cmd_predict = f"nnUNetv2_predict {vstring} -device {device} -d {jdata['dataset_id']} -i {tmpdir+'/input'} -o {tmpdir+'/predicted'} -f {jdata['folds']} -tr {jdata['trainer']} -c {jdata['config']} -p {jdata['plan']} --disable_progress_bar"
    
    sp.call(cmd_predict, shell=True, env=env)
    
    # 4. Run post processing
    print("Starting post-processing")
    
    cmd_process = f"nnUNetv2_apply_postprocessing -i {tmpdir+'/predicted'} -o {tmpdir+'/final'} -pp_pkl_file {nnUNet_results}/{jdata['pp_pkl_file']} -np 8 -plans_json {nnUNet_results}/{jdata['plans_json']}"
    
    sp.call(cmd_process, shell=True, env=env)

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

def install_models():
    my_nnUNet_base = ghost_path() + '/nnUNet'

    env = os.environ
    paths_to_make = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    for p in paths_to_make:
        my_path = os.path.join(my_nnUNet_base, p)
        
        try:
            os.makedirs(my_path)
        except:
            pass

        env[p] = my_path

    # Run the subprocess with modified environment variables
    for model in [237,337,437]:
        print(f"Importing model{model}")
        cmd = f'nnUNetv2_install_pretrained_model_from_zip {my_nnUNet_base}/export{model}.zip'
        if not os.path.exists(f'{my_nnUNet_base}/nnUNet_results/Dataset{model}_UNITY'):
            status = sp.call(shlex.split(cmd), env=env)

def get_model_config(axis):
    D = {
        "axi":
        {
            "pp_pkl_file":"Dataset237_UNITY/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl",
            "plans_json":"Dataset237_UNITY/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json",
            "dataset_id":237,
            "folds":"0 1 2 3 4",
            "trainer":"nnUNetTrainer",
            "config":"3d_fullres",
            "plan":"nnUNetResEncUNetMPlans"
        },
        "sag":
        {
            "pp_pkl_file":"Dataset337_UNITY/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl",
            "plans_json":"Dataset337_UNITY/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json",
            "dataset_id":337,
            "folds":"0 1 2 3 4",
            "trainer":"nnUNetTrainer",
            "config":"3d_fullres",
            "plan":"nnUNetResEncUNetMPlans"
        },
        "cor":
        {
            "pp_pkl_file":"Dataset437_UNITY/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl",
            "plans_json":"Dataset437_UNITY/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json",
            "dataset_id":437,
            "folds":"0 1 2 3 4",
            "trainer":"nnUNetTrainer",
            "config":"3d_fullres",
            "plan":"nnUNetResEncUNetMPlans"
        }
    }

    return D[axis]