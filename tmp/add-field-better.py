#!/usr/bin/env python
""" Append comma to end of all rows missing one field.
As of August 2019, there is a new La Permanence location.  The
scraping script has been modified to add this new field, but the rest
of the file must be updated import.
This is a better version than the 'add-field-dirty.py' which does the
job but without any verification.  Not an issue since task very
simple, but better to do things according to good practices.
"""

import sys
import os
import shutil
import logging
import datetime
import time
from pathlib import Path
# import pandas as pd

logging.basicConfig(level=logging.DEBUG)


FMT_TS = '%Y-%m-%d %H:%M:%S'
BUFFER = 20                     # minimum number of seconds to next minute
DATADIR = Path.home()/'Data'/'la_permanence'
filename = 'availability.csv'
path = DATADIR/filename
newpath = Path.home()/'Projects'/'la_permanence'/'tmp'/'bob.csv'

logging.basicConfig(level=logging.DEBUG)


def enough_time():
    current_time = datetime.datetime.now()
    # logging.debug(current_time.strftime(FMT_TS))
    current_minute = current_time.replace(second=0, microsecond=0)
    next_minute = current_minute + datetime.timedelta(minutes=1)

    to_next_minute = (next_minute - current_time).total_seconds()
    # logging.debug(to_next_minute)
    if to_next_minute > BUFFER:
        print('')
        return current_time
    else:
        print(f"\rTime to next minute: {to_next_minute:.0f}sec...", end='')
        sys.stdout.flush()
        time.sleep(1)
        return enough_time()


def get_last_record():
    with path.open() as f:
        last_line = f.readlines()[-1]

    last_record = \
        datetime.datetime.strptime(
            last_line.split(',')[0],
            FMT_TS
        )
    last_record = last_record.replace(second=0)
    last_record = last_record + datetime.timedelta(hours=1)
    return last_record


def current_minute_recorded():
    last_record = get_last_record()

    current_time = datetime.datetime.now()
    current_minute = current_time.replace(second=0, microsecond=0)

    current_equals_last = current_minute == last_record
    if current_equals_last:
        print('')
        return current_time
    else:
        msg = "\rCurrent minute not yet recorded"
        msg = ' '.join([msg, f"({current_time.strftime(FMT_TS)})..."])
        print(msg, end='')
        sys.stdout.flush()
        time.sleep(1)

        return current_minute_recorded()


def file_correction():
    """Create new file by adding comma at end of rows with missing column."""

    with path.open() as f:
        alllines = f.readlines()

    with newpath.open('w+') as nf:
        firstline = alllines[0].strip() + ',marcadet\n'
        nf.write(firstline)
        for line in alllines[1:]:
            if line.count(',') == 2:
                newline = line.strip() + ',\n'
                nf.write(newline)
            else:
                nf.write(line)
    return newpath


def save_backup():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    backuppath = '-'.join(["BACKUP", path.stem, timestamp]) + path.suffix
    backuppath = DATADIR/backuppath
    shutil.copy(path, backuppath)
    return backuppath


def main():
    print("WARNING")
    print("This script has already been run.")
    return

    save_backup()
    enough_time()
    current_time = current_minute_recorded()
    msg = "Current minute recorded"
    msg = ' '.join([msg, "({current_time.strftime(FMT_TS)})"])
    print(msg)
    newpath = file_correction()
    shutil.copy(newpath, path)


main()
