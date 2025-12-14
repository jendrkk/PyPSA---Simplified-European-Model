"""
Server environment helper: detect and install missing packages into conda.
Falls back to printing exact commands if automation fails.
"""
from __future__ import annotations
import subprocess
import sys
from typing import List

REQUIRED_PACKAGES = [
    "pypsa",
    "pandas",
    "numpy",
    "matplotlib",
]


def _conda_available() -> bool:
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def ensure_packages(packages: List[str] | None = None, env_name: str | None = None) -> None:
    """
    Try to install missing packages with conda; otherwise print commands.
    """
    pkgs = packages or REQUIRED_PACKAGES
    if not _conda_available():
        print("Conda not available. Please run these commands manually:")
        if env_name:
            print(f"conda create -n {env_name} python=3.11 -y")
            print(f"conda activate {env_name}")
        print("conda install -c conda-forge " + " ".join(pkgs))
        return

    # Create env if specified
    if env_name:
        subprocess.run(["conda", "create", "-n", env_name, "python=3.11", "-y"], check=False)
        # Activation must be done in a shell; inform user
        print(f"Please activate the environment: conda activate {env_name}")

    # Install packages
    cmd = ["conda", "install", "-c", "conda-forge"] + pkgs + ["-y"]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    env = sys.argv[1] if len(sys.argv) > 1 else None
    ensure_packages(env_name=env)
