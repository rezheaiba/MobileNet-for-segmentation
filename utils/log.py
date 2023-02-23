import logging
import sys


class logger(object):
    def __init__(self, filename, verbosity=1, name=None):
        return create_logger(filename, verbosity, name)


def create_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

    formatter = logging.Formatter("%(asctime)s-%(levelname)s : %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 屏幕输出
    sh = logging.StreamHandler(sys.stdout)  # 默认是sys.stderr
    sh.setLevel(logging.DEBUG)  # level_dict['0']
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 文件输出
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(level_dict[verbosity])
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
