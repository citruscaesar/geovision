import logging
import datetime
from geovision.io.local import get_new_dir

def get_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    timestamp = datetime.datetime.now().strftime("%d%m%y_%H")
    logging.basicConfig(
        filename = f"{get_new_dir("logs")/timestamp}.log",
        filemode = "a",
        format = "%(asctime)s : %(name)s : %(levelname)s : %(message)s",
        level=logging.INFO
    )
    return logger