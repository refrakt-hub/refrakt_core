"""
Utility functions for dataset directory finding and validation in loaders.
"""

from pathlib import Path
from typing import List, Optional


def _matches_keywords(name: str, keywords: List[str]) -> bool:
    """
    Check if any keyword is present in the directory name (case-insensitive).
    """
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in keywords)


def _search_one_level_deep(base_path: Path, keywords: List[str]) -> Optional[Path]:
    """
    Search one level deep in subdirectories for a directory matching the keywords.
    """
    for item in base_path.iterdir():
        if item.is_dir():
            for subitem in item.iterdir():
                if subitem.is_dir() and _matches_keywords(subitem.name, keywords):
                    return subitem
    return None


def find_directory_by_keywords(
    base_path: Path, keywords: List[str], recursive: bool = True
) -> Optional[Path]:
    """
    Find a directory whose name contains any of the given keywords, optionally searching recursively one level deep.
    Args:
        base_path: Path to search in
        keywords: List of keywords to match in directory names
        recursive: Whether to search one level deep in subdirectories
    Returns:
        Path to the found directory, or None if not found
    """
    # First, look for direct matches
    for item in base_path.iterdir():
        if item.is_dir() and _matches_keywords(item.name, keywords):
            return item
    # Optionally, look deeper in the directory structure
    if recursive:
        return _search_one_level_deep(base_path, keywords)
    return None
