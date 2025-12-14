"""
Core module for building and loading network data.

This module provides functions to construct toy networks and load data from CSV files.
"""

import pandas as pd
from pathlib import Path


def build_network(nodes, edges):
    """
    Build a toy network from nodes and edges.
    
    Args:
        nodes: List of node identifiers (e.g., ["A", "B", "C"])
        edges: List of tuples representing connections (e.g., [("A", "B"), ("B", "C")])
    
    Returns:
        dict: A dictionary representing the network with 'nodes' and 'edges' keys.
    
    Example:
        >>> nodes = ["Node1", "Node2", "Node3"]
        >>> edges = [("Node1", "Node2"), ("Node2", "Node3")]
        >>> network = build_network(nodes, edges)
        >>> print(network)
        {'nodes': ['Node1', 'Node2', 'Node3'], 'edges': [('Node1', 'Node2'), ('Node2', 'Node3')]}
    """
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list")
    if not isinstance(edges, list):
        raise TypeError("edges must be a list")
    
    network = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "num_nodes": len(nodes),
            "num_edges": len(edges)
        }
    }
    return network


def load_csv(path):
    """
    Load a CSV file into a pandas DataFrame with error handling.
    
    Args:
        path: Path to the CSV file (str or Path object)
    
    Returns:
        pd.DataFrame: The loaded data as a DataFrame
    
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        Exception: For other CSV parsing errors
    
    Example:
        >>> # Assuming 'data.csv' exists
        >>> df = load_csv("data.csv")
        >>> print(df.head())
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"CSV file is empty: {path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file {path}: {str(e)}")


# Example usage (for testing/documentation):
if __name__ == "__main__":
    # Example 1: Build a simple network
    nodes = ["Berlin", "Paris", "London"]
    edges = [("Berlin", "Paris"), ("Paris", "London")]
    network = build_network(nodes, edges)
    print("Network:", network)
    
    # Example 2: Load CSV (would need an actual file)
    # df = load_csv("data/sample.csv")
    # print(df.head())
