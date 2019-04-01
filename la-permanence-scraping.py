#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scraping for project on attendance at La Permanence."""

# IMPORTS
import os
import re
#import time
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

HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = SCRIPT_DIR
ARCHIVE_DIR = os.path.join(
    PROJECT_DIR,
    "ARCHIVES"
)
BACKUP_DIR = os.path.join(
    PROJECT_DIR,
    "BACKUP"
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
    foutname = os.path.join(
        PROJECT_DIR,
        "attendance.csv"
    )
    if not os.path.exists(foutname):
        df = pd.DataFrame(columns=["timestamp", "Moulin", "Alésia"])
    else:
        df = pd.read_csv(foutname, sep=SEP)

    # GET PAGE
    url = "https://www.la-permanence.com"
    LOGGER.info("Grabbing page from url: {}...".format(url))
    page = requests.get(url)
    if page.status_code == 200:
        LOGGER.info("Done (status code: {}).".format(page.status_code))
    else:
        LOGGER.warning("Could not download page")
        LOGGER.debug("Status code: {}".format(page.status_code))
        LOGGER.debug("URL: {}.".format(url))
        return None

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
    df.to_csv(foutname, sep=SEP, index=False)

    # CONCLUDING SCRIPT
    end_of_script = time.time()

    LOGGER.info(
        "Time stamp: {0} (duration: {1:.2f} sec)".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_of_script)),
            (end_of_script - run_time)
        )
    )

    LOGGER.info("End of script {}".format(SCRIPT_NAME))


if __name__ == "__main__":
    main()
