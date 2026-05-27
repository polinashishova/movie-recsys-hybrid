"""
Script for downloading models and data required for inference.

Usage example:
cd project
python scripts/download_models.py
"""

from huggingface_hub import snapshot_download
import os
import shutil
from pathlib import Path
import logging
from mrh.utils import load_json


REPO_ID = "shajeless/movie-recsys-hybrid"
ROOT_PATH = Path.cwd()
paths_path = ROOT_PATH / 'configs' / 'paths.json'
paths = load_json(paths_path)
MODEL_DIR = ROOT_PATH / paths["artifacts_dir"] / paths["models_dir"]
PROCESSED_DIR = ROOT_PATH / paths["data_processed_dir"]
FEATURES_DIR = ROOT_PATH / paths["data_features_dir"]

REQUIRED_FILES = {
    'models': ['cb_pipeline.joblib', 'als_model.joblib'],
    'processed': ['cf_user_item_matrix.npz', 'cf_movieId_to_idx.json', 
                  'cf_userId_to_idx.json', 'cb_movieId_to_idx.json', 
                  'neg_cf_movieIds.json'],
    'features': ['cb_features.csv']
}


def check_files_exist() -> tuple[bool, list[str]]:
    """
    Checks whether all required files exist.
    
    Returns:
        tuple[bool, list[str]]: (all_files_exist, list_of_missing_files)
    """
    missing_files = []
    
    for file in REQUIRED_FILES['models']:
        if not (MODEL_DIR / file).exists():
            missing_files.append(f"models/{file}")
    
    for file in REQUIRED_FILES['processed']:
        if not (PROCESSED_DIR / file).exists():
            missing_files.append(f"data/processed/{file}")
    
    for file in REQUIRED_FILES['features']:
        if not (FEATURES_DIR / file).exists():
            missing_files.append(f"data/features/{file}")
    
    return len(missing_files) == 0, missing_files


def download_all() -> None:
    """
    Downloads all files from the repository to the appropriate folders.
    If all files already exist, the download is skipped.
    """
    
    all_exist, missing = check_files_exist()
    
    if all_exist:
        logger.info("All required files already exist. Download skipped.")
        logger.info("Files are located in:")
        logger.info("  - Models: %s", MODEL_DIR)
        logger.info("  - Processed data: %s", PROCESSED_DIR)
        logger.info("  - Features data: %s", FEATURES_DIR)
        return
    
    logger.info("The following files are missing: %s", missing)
    logger.info("Starting download...")
    
    temp_dir = ROOT_PATH / ".temp_download"
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info("Created temporary directory: %s", temp_dir)
        
        logger.info("Starting download from repository %s", REPO_ID)
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=temp_dir
        )
        logger.info("Download from repository completed successfully")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        
        source_models = temp_dir / "models"
        if source_models.exists():
            for file in source_models.glob("*"):
                try:
                    shutil.copy2(file, MODEL_DIR / file.name)
                    logger.info("Copied model: %s", file.name)
                except OSError as e:
                    logger.error("Failed to copy model %s: %s", file.name, e)
                    raise
        else:
            logger.warning("Models directory not found: %s", source_models)
        
        source_processed = temp_dir / "data" / "processed"
        if source_processed.exists():
            for file in source_processed.glob("*"):
                try:
                    shutil.copy2(file, PROCESSED_DIR / file.name)
                    logger.info("Copied processed data: %s", file.name)
                except OSError as e:
                    logger.error("Failed to copy processed data %s: %s", file.name, e)
                    raise
        else:
            logger.warning("Processed data directory not found: %s", source_processed)
        
        source_features = temp_dir / "data" / "features"
        if source_features.exists():
            for file in source_features.glob("*"):
                try:
                    shutil.copy2(file, FEATURES_DIR / file.name)
                    logger.info("Copied features data: %s", file.name)
                except OSError as e:
                    logger.error("Failed to copy features data %s: %s", file.name, e)
                    raise
        else:
            logger.warning("Features data directory not found: %s", source_features)
        
        logger.info("All models and data successfully downloaded!")
        
    except Exception as e:
        logger.exception("Error downloading models and data")
        raise
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary directory deleted: %s", temp_dir)
            except OSError as e:
                logger.warning("Failed to delete temporary directory %s: %s", temp_dir, e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    download_all()