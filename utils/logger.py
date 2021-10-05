# python3.7
"""Utility functions for logging."""

import logging
import os
import sys

__all__ = ['setup_logger']


def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='log'):
    """Sets up logger from target work directory.

    The function will sets up a logger with `DEBUG` log level. Two handlers will
    be added to the logger automatically. One is the `sys.stdout` stream, with
    `INFO` log level, which will print improtant messages on the screen. The other
    is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
    be added time stamp and log level before logged.

    NOTE: If `work_dir` or `logfile_name` is empty, the file stream will be
    skipped.

    Args:
      work_dir: The work directory. All intermediate files will be saved here.
        (default: None)
      logfile_name: Name of the file to save log message. (default: `log.txt`)
      logger_name: Unique name for the logger. (default: `logger`)

    Returns:
      A `logging.Logger` object.

    Raises:
      SystemExit: If the work directory has already existed, of the logger with
        specified name `logger_name` has already existed.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Print log message with `INFO` level or above onto the screen.
        sh = logging.StreamHandler(stream=sys.stdout)
        # sh.setLevel(logging.INFO)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False

    if not work_dir or not logfile_name:
        return logger

    if os.path.exists(work_dir):
        print(f'Work directory `{work_dir}` has already existed!')
    os.makedirs(work_dir, exist_ok=True)

    # Save log message with all levels in log file.
    fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
