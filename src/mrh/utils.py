""" src/mrh/utils.py
Logging setup; loading JSON files; dictionary inversion.
"""

import logging
import json
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


def setup_logging(level: int | str = logging.INFO, log_dir: Path = Path('artifacts/logs'), log_filename: str = 'logs.log'):
    """
    Configure centralized logging with console and file output.
    
    Parameters
    ----------
    level : int | str, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : Path, default=Path('artifacts/logs')
        Directory for storing log files
    log_filename : str, default='logs.log'
        Name of the log file
    
    Returns
    -------
    logging.Logger
        Configured root logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('Logging started. Logs will be written to %s', logfile)

    return logger


JsonType = Union[dict[str, Any], list[Any], str, int, float, bool, None]

def load_json(path: Path) -> JsonType:
    """
    Load JSON data from a file.
    
    Parameters
    ----------
    path : Path
        Path to the JSON file
    
    Returns
    -------
    JsonType
        Parsed JSON content. Can be:
        - dict: JSON object {...}
        - list: JSON array [...]
        - str/int/float/bool/None
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    json.JSONDecodeError
        If the file contains invalid JSON
    OSError
        If the file cannot be read
    """
    logger.info('Loading JSON from %s', path)

    if not path.exists():
        logger.error('JSON file not found: %s', path)
        raise FileNotFoundError(f'File not found: {path}')

    try:
        with path.open('r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            logger.info('JSON successfully loaded from %s', path)
            return data

    except json.JSONDecodeError:
        logger.exception('Invalid JSON format in %s', path)
        raise

    except OSError:
        logger.exception('Failed to read JSON file %s', path)
        raise


def save_json(data: Any, path: Path) -> None:
    """
    Save data to a JSON file.
    
    Parameters
    ----------
    data : Any
        JSON-serializable data: dict, list, str, int, float, bool, None
    path : Path
        Path to save the JSON file
    
    Raises
    ------
    TypeError
        If the data cannot be serialized to JSON
    OSError
        If the file cannot be written
    """
    logger.info('Saving JSON to %s', path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        logger.info('JSON successfully saved to %s', path)

    except TypeError:
        logger.exception('Data cannot be serialized to JSON: %s', path)
        raise

    except OSError:
        logger.exception('Failed to write JSON file %s', path)
        raise


def inverse_dict(dictionary: dict) -> dict:
    """
    Invert a dictionary: swap keys and values.
    
    Parameters
    ----------
    dictionary : dict
        Dictionary to invert. Assumes values are unique and both keys
        and values are immutable types.
    
    Returns
    -------
    dict
        Inverted dictionary where the original dictionary's keys become values
        and values become keys.
    
    Raises
    ------
    TypeError
        If the input parameter is not a dictionary
        If the dictionary values are not hashable
    """
    if not isinstance(dictionary, dict):
        raise TypeError(f"Expected dictionary, got {type(dictionary).__name__}")
    
    result = {}
    for k, v in dictionary.items():
        try:
            result[v] = k
        except TypeError:
            raise TypeError(
                f"Value {v!r} (type {type(v).__name__}) is not hashable "
                "and cannot be a key in the inverted dictionary"
            )
    
    if len(result) != len(dictionary):
        logger.warning(
            "Inverted dictionary has %d keys instead of %d: duplicate values detected",
            len(result), len(dictionary)
        )

    return result
