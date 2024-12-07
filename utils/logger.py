import logging
from datetime import datetime
import os

def get_logger():
    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_file = f'logs/log_{now}.log'
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Stream handler (Optional: コンソール出力用)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_file
