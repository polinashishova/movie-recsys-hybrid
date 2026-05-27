""" src/mrh/data.py
Data handling: downloading, extracting, preparing and preprocessing, saving and loading.
"""

import logging
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.nearest_neighbours import bm25_weight

logger = logging.getLogger(__name__)


def download_data(url: str, path: Path) -> None:
    """
    Download a file from a URL and save it to the specified path.
    
    Parameters
    ----------
    url : str
        URL address of the file to download
    path : Path
        Full path with filename where the file will be saved
    
    Raises
    ------
    urllib.error.URLError
        If the file cannot be downloaded
    OSError
        If the file cannot be saved
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info('File already exists at %s, skipping download', path)
        return

    try:
        logger.info('Downloading %s to %s', url, path)
        urllib.request.urlretrieve(url, path)
        logger.info('Download completed: %s', path)
    except urllib.error.URLError:
        logger.exception('Failed to download from %s', url)
        raise
    except OSError:
        logger.exception('Failed to save file %s', path)
        raise


def extract_archive(
    path_from: Path, 
    directory: Path, 
    expected_path: Path
) -> None:
    """
    Extract a zip archive from path_from to directory if the expected file does not yet exist.
    
    Parameters
    ----------
    path_from : Path
        Path to the zip archive
    directory : Path
        Directory to extract files to
    expected_path : Path
        Path that should exist after successful extraction
    
    Raises
    ------
    ValueError
        If there is an unsafe path in the archive
    """
    directory.mkdir(parents=True, exist_ok=True)

    if expected_path.exists():
        logger.info('Archive already extracted, found %s. Skipping extraction', expected_path)
        return

    with zipfile.ZipFile(path_from, 'r') as zip_ref:
        for member in zip_ref.namelist():
            member_path = (directory / member).resolve()
            if not str(member_path).startswith(str(directory.resolve())):
                raise ValueError(f"Unsafe path in archive: {member}")
        zip_ref.extractall(path=directory)
    
    logger.info('Extraction completed: %s', directory)


def preprocess_tags(tags_string: str | float | None) -> str:
    """
    Preprocess a tag string: convert to lowercase, remove punctuation and extra spaces.
    
    Parameters
    ----------
    tags_string : str, float, or None
        Original tag string. Can be:
        - str: regular string with tags
        - float: typically pd.NA or np.nan
        - None: None value
    
    Returns
    -------
    str
        Processed tag string in lowercase, containing only letters (Latin,
        Cyrillic), digits and spaces. Extra spaces removed.
        If the input string is empty or NaN, returns an empty string.
    
    Raises
    ------
    TypeError
        If the input parameter has an unsupported type
    """
    if not isinstance(tags_string, (str, float, type(None))):
        raise TypeError(
            f"tags_string must be str, float or None, got {type(tags_string).__name__}"
        )
    
    if pd.isna(tags_string) or tags_string == '' or tags_string is None:
        return ''
    
    if isinstance(tags_string, float):
        tags_string = str(tags_string)
    
    tags = tags_string.lower()
    tags = re.sub(r'[^a-zа-я0-9\s]', '', tags)
    tags = ' '.join(tags.split())
    
    return tags


def feature_preparation_cb(
    movies: pd.DataFrame, 
    tags: pd.DataFrame, 
    ratings: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare movie features for the content-based model.
    
    This function creates text features for movies based on genres, release year,
    and user tags. The result is used to build a TF-IDF matrix and subsequent
    similar movie search.
    
    Parameters
    ----------
    movies : pd.DataFrame
        DataFrame with movies, must contain columns 'movieId', 'title', 'genres'
    tags : pd.DataFrame
        DataFrame with tags, must contain columns 'movieId', 'tag'
    ratings : pd.DataFrame
        DataFrame with ratings, must contain column 'movieId'.
        Only movies present in this DataFrame will be included in the result.
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns:
        - year: movie release year (int, 0 if year not determined)
        - genres_fixed: genres as a space-separated string (lowercase)
        - decade: decade (e.g., '1990s', empty string if year = 0)
        - tags: original tags as a string (with duplicates)
        - tags_clean: processed tags (keeping duplicates)
        - genres_decade: combination of genres and decade
        - genres_decade_tags: combination of genres, decade and tags
        
        DataFrame index is movieId.
    
    Raises
    ------
    TypeError
        If movies is not a pandas DataFrame
        If tags is not a pandas DataFrame
        If ratings is not a pandas DataFrame
    ValueError
        If movies is missing columns 'movieId', 'title' or 'genres'
        If tags is missing columns 'movieId' or 'tag'
        If ratings is missing column 'movieId'
        If the movies DataFrame is empty
        If the ratings DataFrame is empty
        If no movies remain after filtering
    """
    if not isinstance(movies, pd.DataFrame):
        raise TypeError(f"movies must be pandas.DataFrame, got {type(movies).__name__}")
    
    if not isinstance(tags, pd.DataFrame):
        raise TypeError(f"tags must be pandas.DataFrame, got {type(tags).__name__}")
    
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(f"ratings must be pandas.DataFrame, got {type(ratings).__name__}")
    
    required_movie_cols = {'movieId', 'title', 'genres'}
    missing_movie_cols = required_movie_cols - set(movies.columns)
    if missing_movie_cols:
        raise ValueError(f"Movies missing columns: {missing_movie_cols}")
    
    if 'movieId' not in tags.columns:
        raise ValueError("Tags missing column 'movieId'")
    if 'tag' not in tags.columns:
        raise ValueError("Tags missing column 'tag'")
    
    if 'movieId' not in ratings.columns:
        raise ValueError("Ratings missing column 'movieId'")
    
    if len(movies) == 0:
        raise ValueError("Movies DataFrame is empty")
    
    if len(ratings) == 0:
        raise ValueError("Ratings DataFrame is empty")
    
    rated_movieIds = ratings['movieId'].unique().tolist()
    
    if len(rated_movieIds) == 0:
        raise ValueError("No unique movieId in ratings")
    
    rated_movies = movies[movies['movieId'].isin(rated_movieIds)].copy()
    
    if len(rated_movies) == 0:
        raise ValueError(
            "No movies from ratings in movies DataFrame. "
            "Check that movieId from ratings exist in movies"
        )
    
    logger.info(
        'Movie filtering: was %d, remaining %d (only user-rated movies)',
        len(movies), len(rated_movies)
    )
    
    processed_movies = rated_movies.copy()
    processed_movies.set_index('movieId', inplace=True)
    
    processed_movies['year'] = processed_movies['title'].str.extract(r'\((\d{4})\)', expand=False)
    processed_movies['year'] = pd.to_numeric(processed_movies['year'], errors='coerce')
    processed_movies['year'] = processed_movies['year'].fillna(0).astype(int)

    processed_movies['genres_fixed'] = processed_movies['genres'].str.replace('|', ' ', regex=False).str.lower().str.strip()

    processed_movies['decade'] = processed_movies['year'].apply(lambda y: f"{(y // 10) * 10}s" if y > 0 else '')

    if len(tags) > 0:
        tags_clean = tags.copy()
        tags_clean['tag'] = tags_clean['tag'].fillna('')

        tags_filtered = tags_clean[tags_clean['movieId'].isin(rated_movieIds)]
        
        if len(tags_filtered) > 0:
            tags_grouped = tags_filtered.groupby('movieId')['tag'].agg(lambda x: ' '.join(x)).to_dict()
            processed_movies['tags'] = processed_movies.index.map(tags_grouped).fillna('')
        else:
            processed_movies['tags'] = ''
            logger.warning('No tags for rated movies')
    else:
        processed_movies['tags'] = ''
        logger.warning('Tags DataFrame is empty')
    
    processed_movies['tags_clean'] = processed_movies['tags'].apply(preprocess_tags)

    processed_movies['genres_decade'] = (processed_movies['genres_fixed'] + ' ' + processed_movies['decade']).str.strip()
    
    processed_movies['genres_decade_tags'] = (processed_movies['genres_decade'] + ' ' + processed_movies['tags_clean']).str.strip()
    
    tags_non_empty = (processed_movies['tags_clean'] != '').sum()
    logger.info(
        'Features prepared: %d movies, of which %d have tags, average tag length %.1f characters',
        len(processed_movies),
        tags_non_empty,
        processed_movies['tags_clean'].str.len().mean() if tags_non_empty > 0 else 0
    )
    
    return processed_movies


def get_ratings_by_threshold(
    ratings: pd.DataFrame, 
    threshold: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter ratings by threshold value. Splits DataFrame into two DataFrames
    based on rating threshold.
    
    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame with ratings, must contain columns 'movieId' and 'rating'
    threshold : float, default=3.0
        Rating threshold value.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of filtered DataFrames with the same columns as the input.
        First with values below the threshold, second with values greater than or equal.
    
    Raises
    ------
    TypeError
        If ratings is not a pandas DataFrame
        If threshold is not a number (int or float)
    ValueError
        If ratings is missing required columns 'movieId' or 'rating'
        If the ratings DataFrame is empty
    """
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(f"ratings must be pandas.DataFrame, got {type(ratings).__name__}")
    
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be int or float, got {type(threshold).__name__}")
    
    required_cols = {'movieId', 'rating'}
    missing_cols = required_cols - set(ratings.columns)
    if missing_cols:
        raise ValueError(f"Ratings missing columns: {missing_cols}")
    
    if len(ratings) == 0:
        raise ValueError("Ratings DataFrame is empty")
    
    min_rating = ratings['rating'].min()
    max_rating = ratings['rating'].max()

    if not (min_rating <= threshold <= max_rating):
        logger.warning(
            'Threshold %.1f is outside rating value range in data [%.1f, %.1f]',
            threshold, min_rating, max_rating
        )
    
    mask = ratings['rating'] >= threshold

    pos_ratings = ratings[mask].copy()
    neg_ratings = ratings[~mask].copy()
    
    logger.info(
        'Filtered %d and %d records with threshold %.1f (total records: %d)',
        len(neg_ratings),
        len(pos_ratings),
        threshold,
        len(ratings)
    )
    
    return (neg_ratings, pos_ratings)


def data_preparation_cf(
    ratings: pd.DataFrame, 
    K1: float = 100, 
    B: float = 0.8
) -> sparse.csr_matrix:
    """
    Prepare data for collaborative filtering with BM25 weighting.
    
    This function creates a user-movie matrix, converting identifiers to indices,
    and applies BM25 weighting to normalize interaction weights.
    
    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame with ratings, must contain columns 'userId', 'movieId'
    K1 : float, default=100
        BM25 K1 parameter, controls weight saturation with repeated interactions
    B : float, default=0.8
        BM25 B parameter, controls length normalization (from 0 to 1)
    
    Returns
    -------
    sparse.csr_matrix
        Sparse user-movie matrix in CSR format with BM25 weights
        Shape: (n_users, n_movies)
    
    Raises
    ------
    TypeError
        If ratings is not a pandas DataFrame
        If K1 or B are not numbers
    ValueError
        If ratings is missing columns 'userId' or 'movieId'
        If the ratings DataFrame is empty
        If no users or movies remain after transformation
        If K1 <= 0 or B is not in range [0, 1]
    """
    
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError(f"ratings must be pandas.DataFrame, got {type(ratings).__name__}")
    
    if not isinstance(K1, (int, float)):
        raise TypeError(f"K1 must be int or float, got {type(K1).__name__}")
    
    if not isinstance(B, (int, float)):
        raise TypeError(f"B must be int or float, got {type(B).__name__}")
    
    if K1 <= 0:
        raise ValueError(f"K1 must be greater than 0, got {K1}")
    
    if not (0 <= B <= 1):
        raise ValueError(f"B must be in range [0, 1], got {B}")
    
    required_cols = {'userId', 'movieId'}
    missing_cols = required_cols - set(ratings.columns)
    if missing_cols:
        raise ValueError(f"Ratings missing columns: {missing_cols}")
    
    if len(ratings) == 0:
        raise ValueError("Ratings DataFrame is empty")
    
    unique_movies = ratings['movieId'].unique()
    unique_users = ratings['userId'].unique()
    
    if len(unique_movies) == 0:
        raise ValueError("No unique movieId in ratings")
    
    if len(unique_users) == 0:
        raise ValueError("No unique userId in ratings")
    
    movieId_to_idx = {int(mid): i for i, mid in enumerate(unique_movies)}
    userId_to_idx = {int(uid): i for i, uid in enumerate(unique_users)}
    
    n_users = len(userId_to_idx)
    n_movies = len(movieId_to_idx)
    
    rows = [userId_to_idx[uid] for uid in ratings['userId']]
    cols = [movieId_to_idx[mid] for mid in ratings['movieId']]
    data = np.ones(len(ratings))
    
    user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    weighted_matrix = sparse.csr_matrix(bm25_weight(user_item_matrix, K1=K1, B=B))
    
    logger.info(
        'Matrix prepared: density = %.4f%%, non-zero elements = %d',
        (weighted_matrix.nnz / (weighted_matrix.shape[0] * weighted_matrix.shape[1])) * 100,
        weighted_matrix.nnz
    )
    
    return weighted_matrix


def save_data(path: Path, data: sparse.spmatrix | pd.DataFrame | np.ndarray) -> None:
    """
    Save data to a file.
    
    This function determines the data type (sparse matrix, DataFrame or NumPy array)
    and saves them to the appropriate format:
    - sparse matrices (csr, csc, coo) -> .npz
    - DataFrame and NumPy arrays -> .csv
    
    Parameters
    ----------
    path : Path
        Path to save the file. If the extension does not match the data type,
        it will be automatically replaced with the correct one (.npz or .csv)
    data : scipy.sparse.spmatrix or pd.DataFrame or np.ndarray
        Data to save. Supports SciPy sparse matrices,
        pandas DataFrame and NumPy arrays
    
    Returns
    -------
    None
    
    Raises
    ------
    OSError
        If the file cannot be saved (write error, insufficient permissions, etc.)
    TypeError
        If the data type is not supported (not sparse matrix, DataFrame or ndarray)
    """
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        logger.info('File %s already exists, skipping save', path)
        return
    
    if sparse.isspmatrix(data):
        if path.suffix != '.npz':
            new_path = path.with_suffix('.npz')
            logger.info('Extension changed from %s to .npz for sparse matrix', path.suffix)
            path = new_path
        
        try:
            logger.info('Saving sparse matrix to %s', path)
            sparse.save_npz(path, data)
            logger.info('Sparse matrix successfully saved')
            return
        except OSError as e:
            logger.exception('Error saving sparse matrix to %s', path)
            raise OSError(f"Failed to save sparse matrix: {e}") from e
    
    if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
        if path.suffix != '.csv':
            new_path = path.with_suffix('.csv')
            logger.info('Extension changed from %s to .csv for tabular data', path.suffix)
            path = new_path
        
        try:
            logger.info('Saving tabular data to %s', path)
            if isinstance(data, np.ndarray):
                logger.debug('Converting NumPy array to DataFrame')
                data = pd.DataFrame(data)
            data.to_csv(path, sep=',', index=False, encoding='utf-8')
            logger.info('Tabular data successfully saved: %d rows, %d columns', 
                       len(data), len(data.columns))
            return
        except OSError as e:
            logger.exception('Error saving tabular data to %s', path)
            raise OSError(f"Failed to save tabular data: {e}") from e
    
    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        "Expected: scipy.sparse.spmatrix, pandas.DataFrame or numpy.ndarray"
    )


def load_data(path: Path) -> sparse.spmatrix | pd.DataFrame:
    """
    Load data from a file with automatic format detection.
    
    This function determines the file format by extension and loads the data
    into the appropriate type:
    - .npz -> SciPy sparse matrix
    - .csv -> pandas DataFrame
    
    Parameters
    ----------
    path : Path
        Path to the file to load. Supports .npz and .csv extensions
    
    Returns
    -------
    scipy.sparse.spmatrix or pd.DataFrame
        Loaded data:
        - for .npz: sparse matrix (csr_matrix)
        - for .csv: pandas DataFrame
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path
    ValueError
        If the file extension is not supported (.npz or .csv)
    OSError
        If the file cannot be read (corrupted, insufficient permissions, etc.)
    """
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info('Loading data from %s', path)
    
    if path.suffix == '.npz':
        try:
            data = sparse.load_npz(path)
            logger.info(
                'Loaded sparse matrix: shape = %s, non-zero elements = %d, density = %.4f%%',
                data.shape,
                data.nnz,
                (data.nnz / (data.shape[0] * data.shape[1])) * 100 if data.shape[0] * data.shape[1] > 0 else 0
            )
            return data
        except Exception as e:
            logger.exception('Error loading sparse matrix from %s', path)
            raise OSError(f"Failed to load sparse matrix: {e}") from e
    
    if path.suffix == '.csv':
        try:
            data = pd.read_csv(path, encoding='utf-8')
            logger.info(
                'Loaded DataFrame: shape = %s, columns = %s',
                data.shape,
                list(data.columns)
            )
            return data
        except Exception as e:
            logger.exception('Error loading CSV from %s', path)
            raise OSError(f"Failed to load CSV file: {e}") from e
    
    raise ValueError(
        f"Unsupported file format: {path.suffix}. "
        "Expected: .npz (sparse matrix) or .csv (DataFrame)"
    )
