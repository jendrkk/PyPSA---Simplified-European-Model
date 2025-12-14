"""
Utility functions for file I/O and directory management.

This module provides helper functions for common operations like
directory creation and JSON file handling.
"""

import json
from pathlib import Path


def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory (str or Path object)
    
    Returns:
        Path: The Path object for the directory
    
    Example:
        >>> ensure_dir("data/processed")
        PosixPath('data/processed')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    """
    Read a JSON file and return its contents.
    
    Args:
        path: Path to the JSON file (str or Path object)
    
    Returns:
        dict or list: The parsed JSON content
    
    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    
    Example:
        >>> data = read_json("config.json")
        >>> print(data)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {path}: {e.msg}", e.doc, e.pos)


def write_json(obj, path):
    """
    Write an object to a JSON file.
    
    Args:
        obj: The object to serialize (must be JSON-serializable)
        path: Path to the output JSON file (str or Path object)
    
    Returns:
        Path: The Path object for the written file
    
    Example:
        >>> data = {"name": "test", "value": 42}
        >>> write_json(data, "output.json")
        PosixPath('output.json')
    """
    path = Path(path)
    
    # Ensure parent directory exists
    if path.parent != Path('.'):
        ensure_dir(path.parent)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return path
    except TypeError as e:
        raise TypeError(f"Object is not JSON-serializable: {e}")


# Example usage (for testing/documentation):
if __name__ == "__main__":
    # Example 1: Ensure directory exists
    data_dir = ensure_dir("data/temp")
    print(f"Directory created/verified: {data_dir}")
    
    # Example 2: Write and read JSON
    sample_data = {
        "nodes": ["A", "B", "C"],
        "edges": [["A", "B"], ["B", "C"]],
        "config": {"version": "1.0"}
    }
    
    json_path = data_dir / "sample.json"
    write_json(sample_data, json_path)
    print(f"Written to: {json_path}")
    
    loaded_data = read_json(json_path)
    print(f"Loaded data: {loaded_data}")
