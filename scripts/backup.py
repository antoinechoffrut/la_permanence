#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hourly, daily and weekly backup of file with dataset for La
Permanence project.
"""

# IMPORTS
import os
# import sys
import re
import shutil
import datetime
import pytz
import logging
# import requests
# from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

# GLOBAL VARIABLES
SEP = ","                                # separator for .csv file
SCRIPT_NAME = os.path.basename(__file__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# SCRIPT_NAME = 'backup.py'
# SCRIPT_DIR = os.path.join(
#     os.path.expanduser('~'),
#     'Projects',
#     'la_permanence',
#     'scripts'
# )
DATA_DIR = os.path.join(os.path.expanduser('~'), 'Data', 'la_permanence')

DATAFILE_NAME = 'availability.csv'  # formerly: 'attendance.csv'
DATAFILE_PATH = os.path.join(
    DATA_DIR,
    DATAFILE_NAME
)

# LOGGING
SCRIPT_LOG = re.sub(".py$", ".log", SCRIPT_NAME)
SCRIPT_LOG = os.path.join(SCRIPT_DIR, SCRIPT_LOG)
LOGGER = logging.getLogger("script_logger")   # __name__
LOGGER.setLevel(logging.DEBUG)
FILE_HANDLER = logging.FileHandler(SCRIPT_LOG)
FILE_FORMATTER = logging.Formatter(
    '%(asctime)s - %(levelname)-10s: %(message)s'
)
FILE_HANDLER.setFormatter(FILE_FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

# TIMEZONES
TZ_UTC = pytz.timezone("UTC")

# DATETIME FORMATS
FMT_TS = '%Y-%m-%d %H:%M:%S'    # for timestamps
# FMT_FN = '%Y-%m-%d-%H-%M-%S'    # for filenames
# FMT_LOG = '%a %d %b %Y %H:%M:%S %Z%z'  # for logging


# CLASS DEFINITIONS

# FUNCTION DEFINITIONS
def dateparse(str):
    return pd.datetime.strptime(str, FMT_TS)


def ts_roundoff(ts, freq='1H'):
    """Convert timetamp to UTC standard and return round off."""
    ts = ts.tz_convert(TZ_UTC)
    result = ts
    if freq == '1H':
        result = pd.Timestamp(
            ts.year,
            ts.month,
            ts.day,
            ts.hour
        )
    elif freq == '1D':
        result = pd.Timestamp(
            ts.year,
            ts.month,
            ts.day
        )
    elif freq == '1W':
        result = ts
        res_str = ts.strftime('%G-W%V-1')  # this week's first day
        res_dt = datetime.datetime.strptime(res_str, ('%G-W%V-%u'))
        res_ts = \
            pd.Timestamp(
                res_dt.year,
                res_dt.month,
                res_dt.day
            ).tz_localize(TZ_UTC)
        result = res_ts

    return result


def is_time_to_backup(run_time, last_ts, freq='1H'):
    """Return True if it is time to backup dataset"""
    if freq == '1W':
        freq = '7D'

    return ts_roundoff(run_time, freq) \
        >= ts_roundoff(last_ts, freq) + pd.Timedelta(freq)


def load_data(filepath):
    df = pd.read_csv(
        filepath,
        sep=SEP,
        parse_dates=['timestamp'],
        date_parser=dateparse
    )

    df['timestamp'] = \
        df['timestamp'].apply(lambda dt: dt.tz_localize(TZ_UTC))

    return df


def get_run_time():
    """Return timestamp of current time and date."""
    now = datetime.datetime.now(tz=TZ_UTC)
    run_time = pd.Timestamp(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second
    ).tz_localize(TZ_UTC)

    return run_time


def get_backup_path(mode):
    backup_name = ".".join([
        "-".join([
            ".".join(DATAFILE_NAME.split('.')[0:-1]),
            "BACKUP",
            mode
        ]),
        DATAFILE_NAME.split('.')[-1]
    ])
    backup_path = os.path.join(DATA_DIR, backup_name)
    return backup_name, backup_path


def main():
    """Get number of available seats at La Permanence coworking
    spaces."""
    LOGGER.info("---------- BEGIN ----------")

    if not os.path.exists(DATAFILE_PATH):
        LOGGER.error(
            f"Datafile {DATAFILE_NAME} does not exist."
        )
        LOGGER.info("---------- END ----------")
        return

    run_time = get_run_time()

    for (freq, mode) in [('1H', 'hourly'), ('1D', 'daily'), ('1W', 'weekly')]:
        # LOGGER.info(f"Backup mode: {mode}.")

        backup_name, backup_path = get_backup_path(mode)

        if not os.path.exists(backup_path):
            shutil.copy(DATAFILE_PATH, backup_path)
            msg = ' '.join([
                f"Backup file {backup_name} does not exist,",
                f"copied from {DATAFILE_NAME}"
                ])
            LOGGER.info(msg)
        else:
            df_backup = load_data(backup_path)

            last_ts = df_backup['timestamp'].max()

            if is_time_to_backup(run_time, last_ts, freq):
                LOGGER.info(f"Mode {mode}: backup file updated.")
                shutil.copy(DATAFILE_PATH, backup_path)
            else:
                LOGGER.info(f"Mode {mode.ljust(10)}: not yet time to update.")


    LOGGER.info("---------- END ----------")


if __name__ == "__main__":
    main()
