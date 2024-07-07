import logging
from datetime import datetime
import os

# Set up the main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all types of log messages
log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_processing.log')

# Ensure the directory exists
log_dir = 'scripts/data/logs'
os.makedirs(log_dir, exist_ok=True)

# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set up a file handler
file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
file_handler.setLevel(logging.DEBUG)  # Capture all messages to the file
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Set up a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Capture info and above messages to the console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
