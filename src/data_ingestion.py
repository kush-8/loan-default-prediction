# src/data_ingestion.py
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

# Set up a simple logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_data():
    """
    Downloads the Home Credit Default Risk dataset from Kaggle and extracts it.
    """
    competition_name = 'home-credit-default-risk'
    download_path = 'data/raw'
    
    # Ensure the target directory exists
    os.makedirs(download_path, exist_ok=True)
    
    logging.info("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    logging.info(f"Downloading dataset for competition: {competition_name}...")
    # This command downloads the competition files into the specified path
    api.competition_download_files(competition_name, path=download_path, quiet=False)
    
    logging.info("Extracting files...")
    # The Kaggle API often zips all competition files into one archive
    zip_file_path = os.path.join(download_path, f"{competition_name}.zip")
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        logging.info("Extraction complete.")
        # Clean up the zip file after extraction
        os.remove(zip_file_path)
        logging.info(f"Removed zip file: {zip_file_path}")
    except FileNotFoundError:
        logging.error(f"Error: The zip file {zip_file_path} was not found. The download may have failed or the file name is different.")
    except Exception as e:
        logging.error(f"An error occurred during extraction: {e}")

if __name__ == '__main__':
    download_and_extract_data()