#!/usr/bin/env python
"""Collect number of available seats at the La Permanence coworking
spaces from https://www.la-permanence.com and write data into csv
file.
"""


import os
import sys
import re
import logging
MODULES_DIR = \
    os.path.join(os.path.expanduser("~"), "Projects/la_permanence/modules")
sys.path.insert(0, MODULES_DIR)
import LaPermanence as lap


if __name__ == '__main__':
    script_name = os.path.basename(__file__)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_log = os.path.join(
        script_dir,
        re.sub(".py$", ".log", script_name)
    )

    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d | %(levelname)-10s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                script_log,
                mode='+a'
            ),
            logging.StreamHandler()
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    lap.scrape()
