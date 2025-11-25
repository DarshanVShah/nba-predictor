"""
Utility to download required data and model files if they don't exist.
This solves the Git LFS and volume upload issues.
"""
import os
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL to a destination path.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        chunk_size: Chunk size for streaming download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {destination}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with streaming for large files
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (chunk_size * 100) == 0:  # Log every 100 chunks
                            logger.info(f"Downloaded {percent:.1f}% ({downloaded}/{total_size} bytes)")
        
        logger.info(f"✅ Successfully downloaded {destination} ({downloaded} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading {url}: {str(e)}")
        # Clean up partial file
        if os.path.exists(destination):
            try:
                os.remove(destination)
            except:
                pass
        return False

def ensure_files_exist(
    final_dataset_url: str = None,
    processed_data_url: str = None,
    model_url: str = None,
    predictors_url: str = None,
    scaler_url: str = None
):
    """
    Ensure required files exist, downloading them if necessary.
    
    Args:
        final_dataset_url: URL for final_dataset.csv
        processed_data_url: URL for processed_nba_data.csv
        model_url: URL for nba_model.pkl
        predictors_url: URL for predictors.pkl
        scaler_url: URL for scaler.pkl
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'backend', 'data', 'processed')
    models_dir = os.path.join(base_dir, 'backend', 'models')
    
    files_to_check = [
        (os.path.join(data_dir, 'final_dataset.csv'), final_dataset_url, 'final_dataset.csv'),
        (os.path.join(data_dir, 'processed_nba_data.csv'), processed_data_url, 'processed_nba_data.csv'),
        (os.path.join(models_dir, 'nba_model.pkl'), model_url, 'nba_model.pkl'),
        (os.path.join(models_dir, 'predictors.pkl'), predictors_url, 'predictors.pkl'),
        (os.path.join(models_dir, 'scaler.pkl'), scaler_url, 'scaler.pkl'),
    ]
    
    for file_path, url, name in files_to_check:
        if url:  # Only download if URL is provided
            if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:  # File missing or too small (pointer file)
                logger.info(f"File {name} missing or too small, downloading from {url}")
                download_file(url, file_path)
            else:
                file_size = os.path.getsize(file_path)
                logger.info(f"✅ File {name} already exists ({file_size:,} bytes)")
        else:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"✅ File {name} exists ({file_size:,} bytes)")
            else:
                logger.warning(f"⚠️ File {name} not found and no download URL provided")




