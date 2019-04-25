#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect number of available seats at the La Permanence coworking
spaces from https://www.la-permanence.com and write data into csv
file.
"""

# IMPORTS
import os
# import sys
import re
# import shutil
import datetime
import pytz
import logging
import requests
from bs4 import BeautifulSoup
# import numpy as np
# import pandas as pd

# GLOBAL VARIABLES
SEP = ','

SCRIPT_NAME = os.path.basename(__file__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(os.path.expanduser('~'), 'Data', 'la_permanence')
FILENAME = 'availability.csv'  # 'aaaa.csv'
FILEPATH = os.path.join(DATA_DIR, FILENAME)


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


# DATETIME FORMATS
FMT_TS = '%Y-%m-%d %H:%M:%S'    # for timestamps

# CLASS DEFINITIONS


# FUNCTION DEFINITIONS


def get_timestamp():
    """Return timestamp of data collection in format YYYY-MM-DD hh:mm"""
    tz_utc = pytz.timezone("UTC")
    run_time = datetime.datetime.now(tz=tz_utc)
    timestamp = run_time.strftime(FMT_TS)
    return timestamp


def get_page():
    """Retrieve page http://www.la-permanence.com"""
    url = "https://www.la-permanence.com"  # "http://bsegerglb.com"
    try:
        page = requests.get(url)
    except requests.ConnectionError as e:
        page = None
        LOGGER.error(e)
    return page


def extract_data(page):
    """Extract number of seats at Moulin and Alésia coworking spaces
    from html page."""

    soup = BeautifulSoup(page.text, "html.parser")
    locations = soup.find_all(
        "div",
        {"class": "seats"}
    )

    moulin_seats = ""  # np.NaN
    alesia_seats = ""  # np.NaN

    for location in locations:
        if "Moulin" in location.find_all("p")[0].text:
            moulin_seats = re.sub(
                "Places",
                "",
                location.find_all("span")[0].text
            ).strip()
        elif "Alésia" in location.find_all("p")[0].text:
            alesia_seats = re.sub(
                "Places",
                "",
                location.find_all("span")[0].text
            ).strip()

    return moulin_seats, alesia_seats


def check_path():
    """Check existence of directory to contain csv file."""
    result = 0
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(FILEPATH):
        result += 1
        with open(FILEPATH, 'w') as file:
            file.write('timestamp,moulin,alesia')
            file.write('\n')
    return result


def write_data(line, filename=FILEPATH):
    with open(FILEPATH, 'a+') as file:
        file.write(line)
        file.write('\n')
    return


# MAIN FUNCTION
def main():
    """Record timestamp, scrape web page, write into csv file."""
    LOGGER.info("---------- BEGIN ----------")
    LOGGER.info("Record timestamp.")
    timestamp = get_timestamp()
    # print(f'Timestamp: {timestamp}')

    LOGGER.info("Retrieve web page.")
    page = get_page()

    if page is None:
        return

    LOGGER.info("Extract number of seats.")
    moulin_seats, alesia_seats = extract_data(page)
    # print(f'Moulin: {moulin_seats}.  Alésia: {alesia_seats}')

    line = SEP.join([timestamp, moulin_seats, alesia_seats])
    # LOGGER.info(f"Line to append to csv file: {line}")

    LOGGER.info("Create csv file if does not exist.")
    check_path()
    LOGGER.info(f'Write "{line}" to end of file.')
    write_data(line)
    LOGGER.info("---------- END ----------")


if __name__ == "__main__":
    main()
