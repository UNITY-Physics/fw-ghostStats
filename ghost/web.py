import os
import requests
from tqdm import tqdm

GHOST_ASSETS = {
    "figshare":{
        "PhantomModels":{
            "Caliber137":{
                "object_id":"26954638",
                "version":1
            }
        },
        "nnUNet":{
            "Caliber137_fiducials":{
                "object_id":"26892781",
                "version":1
            }
        },
        "ExampleData":{
            "UNITY_QA":{
                "object_id":"26954056",
                "version":1
            }
        }
    }
}


def figshare_download(object_id, version, file_path, over_write=False):
    article = requests.get(f"https://api.figshare.com/v2/articles/{object_id}/versions/{version}").json()
    for file in article['files']:
        fname = os.path.join(file_path, file['name'])
        if (not os.path.exists(fname)) or over_write:
            download_file(file['download_url'], fname)
        else:
            print(f"Already downloaded: {fname}")


def download_file(url, fname):
    # Send a HTTP request to get the content length (i.e., file size)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # Total size in bytes

    # Check output path
    file_path = os.path.dirname(fname)
    if not os.path.exists(file_path):
        print(f"Output path does not exist. Creating it\n{file_path}")
        os.makedirs(file_path)

    # Open the file in binary write mode
    print(f"Starting download from: {url}")
    print(f"Total file size: {total_size/ (1024*1024):.1f} MB")
    print(f"Writing to: {fname}")
    
    with open(fname, 'wb') as f:
    
        for chunk in tqdm(response.iter_content(chunk_size=512 * 1024), 
                          total=total_size // (512 * 1024), 
                          unit='KB', 
                          unit_scale=True):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)
    
    return True