import os
import requests
import tarfile

def download_and_extract(url, output_path):
    """
    Downloads a tar.gz file from the given URL and extracts it into output_path.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Determine local file name
    local_filename = os.path.join(output_path, os.path.basename(url))

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract the tar.gz
    with tarfile.open(local_filename, 'r:gz') as tar:
        tar.extractall(path=output_path)

    print(f"Downloaded and extracted to {output_path}")