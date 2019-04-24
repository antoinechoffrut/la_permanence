#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scraping for project on attendance at La Permanence."""

# IMPORTS
import os
import sys
import re
import shutil
import datetime
import pytz
import logging
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

# GLOBAL VARIABLES
SEP = ","                                # separator for .csv file
SCRIPT_NAME = os.path.basename(__file__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


PROJECT_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects', 'la-permanence'
)  # SCRIPT_DIR
BACKUP_DIR = os.path.join(PROJECT_DIR, 'DATA_BACKUPS')
DATA_DIR = os.path.join(os.path.expanduser('~'), 'Data', 'la-permanence')

DATAFILE_NAME = 'attendance.csv'
DATAFILE_PATH = os.path.join(
    DATA_DIR,
    DATAFILE_NAME  # "attendance.csv"
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
FMT_TS = '%Y-%m-%d %H:%M'    # for timestamps
FMT_FN = '%Y-%m-%d-%H-%M-%S'    # for filenames
FMT_LOG = '%a %d %b %Y %H:%M:%S %Z%z'  # for logging


# CLASS DEFINITIONS

# FUNCTION DEFINITIONS
def dateparse(str):
    return pd.datetime.strptime(str, '%Y-%m-%d-%H-%M-%S')


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


def is_time_to_backup(run_time_ts, last_ts, freq='1H'):
    """Return True if it is time to backup dataset"""
    if freq == '1W':
        freq = '7D'

    return ts_roundoff(run_time_ts, freq) \
        >= ts_roundoff(last_ts, freq) + pd.Timedelta(freq)


def main():
    """Get number of available places at La Permanence (two locations
    in Paris)."""
    # Clock IN
    # run_time = TZ_UTC.localize(datetime.datetime.now())
    run_time = datetime.datetime.now(tz=TZ_UTC)

    timestamp = run_time.strftime(FMT_FN)

    os.system("clear")
    LOGGER.info("-"*10)
    LOGGER.info(run_time.strftime(FMT_LOG))
    LOGGER.debug("Process id: {}".format(os.getpid()))
    LOGGER.info("Running script {}".format(SCRIPT_NAME))
    LOGGER.debug("Directory: {}".format(SCRIPT_DIR))
    LOGGER.info("Logfile: {}".format(SCRIPT_LOG))

    # GET DATAFRAME FROM CSV FILE - IF EXISTS, OR ELSE CREATE
    LOGGER.info("Datafile: {0}".format(DATAFILE_PATH))
    LOGGER.info("Backup files in: {0}".format(BACKUP_DIR))

    if not os.path.exists(DATAFILE_PATH):
        LOGGER.info("Datafile {0} does not exist".format(DATAFILE_NAME))
        df = pd.DataFrame(columns=["timestamp", "Moulin", "Alésia"])
    else:
        LOGGER.info("Datafile: {0} already exists".format(DATAFILE_NAME))
        df = pd.read_csv(DATAFILE_PATH, sep=SEP)

    # GET PAGE
    url = "https://www.la-permanence.com"
    LOGGER.info("Grabbing page from url: {}...".format(url))
    try:
        page = requests.get(url)
        LOGGER.info("Done (status code: {}).".format(page.status_code))
    except requests.ConnectionError as e:
        LOGGER.error("Error connecting to url {}".format(url))
        LOGGER.error(e)
        sys.exit(1)

    # if page.status_code == 200:
    #     LOGGER.info("Done (status code: {}).".format(page.status_code))
    # else:
    #     LOGGER.warning("Could not download page")
    #     LOGGER.debug("Status code: {}".format(page.status_code))
    #     LOGGER.debug("URL: {}.".format(url))
    #    return None

    soup = BeautifulSoup(page.text, "html.parser")
    locations = soup.find_all(
        "div",
        {"class": "seats"}
    )

    moulin_seats = np.NaN
    Alesia_seats = np.NaN

    for location in locations:
        if "Moulin" in location.find_all("p")[0].text:
            moulin_seats = re.sub(
                "Places",
                "",
                location.find_all("span")[0].text
            ).strip()
        elif "Alésia" in location.find_all("p")[0].text:
            Alesia_seats = re.sub(
                "Places",
                "",
                location.find_all("span")[0].text
            ).strip()
        else:
            LOGGER.warning("No data found for either location.")

    df = df.append(
        {
            "timestamp": timestamp,
            "Moulin": moulin_seats,
            "Alésia": Alesia_seats
        },
        ignore_index=True
    )

    # WRITE DATAFRAME INTO CSV FILE
    LOGGER.info("Writing data to: {0}".format(DATAFILE_NAME))
    df.to_csv(DATAFILE_PATH, sep=SEP, index=False)

    # BACKUP FILE - IF TIME TO DO SO
    LOGGER.info(
        "Check whether backup files need updating."
        )
    run_time_ts = pd.Timestamp(
        run_time.year,
        run_time.month,
        run_time.day,
        run_time.hour,
        run_time.minute,
        run_time.second
    ).tz_localize(TZ_UTC)

    for (freq, mode) in [('1H', 'hourly'), ('1D', 'daily'), ('1W', 'weekly')]:
        LOGGER.info("Backup mode: {0}".format(mode))

        backup_name = ".".join([
            "-".join([
                ".".join(DATAFILE_NAME.split('.')[0:-1]),
                "BACKUP",
                mode
            ]),
            DATAFILE_NAME.split('.')[-1]
        ])
        backup_path = os.path.join(BACKUP_DIR, backup_name)

        if not os.path.exists(backup_path):
            shutil.copy(DATAFILE_PATH, backup_path)
            LOGGER.info(
                "Backup file {0} did not exist, now created.".format(
                    backup_name
                )
            )
        else:

            df_backup = pd.read_csv(
                backup_path,
                sep=',',
                dtype={'Moulin': np.uint8, 'Alésia': np.uint8},
                parse_dates=['timestamp'],
                date_parser=dateparse
            )
            df_backup['timestamp'] = \
                df_backup['timestamp'].apply(
                    lambda ts: ts.tz_localize(TZ_UTC)
                    )

            last_ts = df_backup['timestamp'].max()

            if is_time_to_backup(run_time_ts, last_ts, freq):
                LOGGER.info("Time to update {0}".format(backup_name))
                shutil.copy(DATAFILE_PATH, backup_path)
            else:
                LOGGER.info("Not yet time to update {0}".format(backup_name))




    # CONCLUDING SCRIPT

    end_of_script = datetime.datetime.now(tz=TZ_UTC)

    LOGGER.info(
        "End of script: {0}".format(
            end_of_script.strftime(FMT_LOG)
        )
    )
    # LOGGER.info(
    #     "Time stamp: {0} (duration: {1:.2f} sec)".format(
    #         datetime.strftime("%Y-%m-%d %H:%M:%S", datetime.localtime(end_of_script)),
    #         (end_of_script - run_time)
    #     )
    # )

    LOGGER.info("End of script {}".format(SCRIPT_NAME))


if __name__ == "__main__":
    main()
