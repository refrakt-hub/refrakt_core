#!/usr/bin/env python3
"""
Development setup script for refrakt_core.

This script helps set up the development environment with all necessary tools.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up refrakt_core development environment...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install dev dependencies
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        "Installing development dependencies"
    ):
        sys.exit(1)
    
    # Install pre-commit hooks
    if not run_command(
        ["pre-commit", "install"],
        "Installing pre-commit hooks"
    ):
        print("⚠️  Pre-commit installation failed, but continuing...")
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    
    tools = [
        ("pytest", "Testing framework"),
        ("black", "Code formatter"),
        ("ruff", "Linter"),
        ("isort", "Import sorter"),
        ("mypy", "Type checker"),
        ("pre-commit", "Git hooks"),
    ]
    
    all_good = True
    for tool, description in tools:
        try:
            subprocess.run([tool, "--version"], check=True, capture_output=True)
            print(f"✅ {tool} ({description})")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {tool} ({description}) - not found")
            all_good = False
    
    if all_good:
        print("\n🎉 Development environment setup complete!")
        print("\n📋 Available commands:")
        print("  pytest          - Run tests")
        print("  black .         - Format code")
        print("  ruff check .    - Lint code")
        print("  isort .         - Sort imports")
        print("  mypy src/       - Type check")
        print("  pre-commit run --all-files  - Run all pre-commit hooks")
    else:
        print("\n⚠️  Some tools failed to install. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 