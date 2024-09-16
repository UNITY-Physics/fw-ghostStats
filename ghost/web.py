import requests
from tqdm import tqdm


def figshare_download(object_id, version):
    article = requests.get(f"https://api.figshare.com/v2/articles/{object_id}/versions/{version}").json()
    for file in article['files']:
        download_file(file['download_url'], file['name'])


def download_file(url, fname):
    # Send a HTTP request to get the content length (i.e., file size)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
    
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