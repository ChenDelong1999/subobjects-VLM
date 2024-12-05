import os
import datasets
import concurrent.futures
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO
import random

def download_image(args):
    index, image_url, image_folder = args
    print(args)
    filename = os.path.join(image_folder, f"{index}.jpg")
    if os.path.exists(filename):
        # Skip if already downloaded
        return
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    try:
        response = session.get(image_url, timeout=10)
        if response.status_code == 200:
            try:
                image = Image.open(BytesIO(response.content))
                image.convert('RGB').save(filename, format='JPEG')
            except Exception as e:
                print(f"Error processing image from {image_url}: {e}")
        else:
            print(f"Failed to download {image_url}, status code {response.status_code}")
    except Exception as e:
        print(f"Exception while downloading {image_url}: {e}")

def main():
    data = datasets.load_dataset("allenai/pixmo-cap", split="train")
    total_samples = len(data)
    image_folder = '/private/home/delong/workspace/data/pixmo-cap'
    os.makedirs(image_folder, exist_ok=True)
    
    samples = [(i, data[i]['image_url'], image_folder) for i in range(total_samples)]
    random.shuffle(samples)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(download_image, samples), total=total_samples))

if __name__ == '__main__':
    main()
