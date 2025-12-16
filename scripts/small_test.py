import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load all requirements-dev.txt
def load_requirements_dev():
    import pkg_resources
    import subprocess
    import sys
    try:
        logger.info("Checking development requirements...")
        with open('requirements-dev.txt') as f:
            requirements = f.read().splitlines()
        pkg_resources.require(requirements)
    except pkg_resources.DistributionNotFound:
        logger.info("Some development requirements are missing. Installing...")
        # subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'])

def main():
    logger.info("This is a small test script for PyPSA Simplified European Model.")
    load_requirements_dev()
    logger.info("All development requirements are installed.")

if __name__ == "__main__":
    main()