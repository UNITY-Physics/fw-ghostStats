import json
import pydicom
import os
import re
import shutil

#### Append temperature to json file ####
def find_first_dicom_file(p):
    found_dcm = False
    dpath = p
    i = 0

    while not found_dcm:
        if os.path.isfile(dpath):
            try:
                dcm = pydicom.dcmread(dpath)
                found_dcm = True
                return dpath
            except:
                print('Non dicom file in path')
        else:
            dpath = os.path.join(dpath,os.listdir(dpath)[-1])

def get_temperature(dcm_file):
    d = pydicom.dcmread(dcm_file)
    temp = int(re.findall(r'[0-9]{2}', d[0x0010,0x4000].value)[0])
    return temp

def update_json(jf, temp):
    with open(jf, 'r') as f:
        d = json.load(f)

    d['Temperature'] = temp

    with open(jf+'_tmp', 'w') as f:
        json.dump(d, f, indent = 4)

    os.remove(jf)
    shutil.move(jf+'_tmp', jf)

###### Example of how to run ######

# Finds a dicom file in a given folder. For longitudinal study this should be at session level
dicom_folder = '/Users/emil/Box_Lund/UNITY_Pilot_QA_Study/sourcedata/LUND'
dcm_file = find_first_dicom_file(dicom_folder)
temp = get_temperature(dcm_file)

# Here there should be a function to automatically go through all the json files in a given session. Probably easy with pybids?
jf = '/Users/emil/Library/CloudStorage/Box-Box/UNITY_Pilot_QA_Study/rawdata/sub-LUND/ses-01/anat/sub-LUND_ses-01_rec-axi_flair.json'
update_json(jf, temp)