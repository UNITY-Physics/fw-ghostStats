import os
import requests
from tqdm import tqdm


def figshare_download(object_id, version, file_path):
    article = requests.get(f"https://api.figshare.com/v2/articles/{object_id}/versions/{version}").json()
    for file in article['files']:
        download_file(file['download_url'], file['name'], file_path)


def download_file(url, fname, file_path):
    # Send a HTTP request to get the content length (i.e., file size)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # Total size in bytes

    # Check output path
    if not os.path.exists(file_path):
        print(f"Output path does not exist. Creating it\n{file_path}")
        os.makedirs(file_path)

    out_name = os.path.join(file_path, fname)
    # Open the file in binary write mode
    print(f"Starting download from: {url}")
    print(f"Total file size: {total_size/ (1024*1024):.1f} MB")
    print(f"Writing to: {out_name}")
    
    with open(out_name, 'wb') as f:
    
        for chunk in tqdm(response.iter_content(chunk_size=512 * 1024), 
                          total=total_size // (512 * 1024), 
                          unit='KB', 
                          unit_scale=True):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)
    
    return True