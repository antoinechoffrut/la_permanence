#!/usr/bin/env python
"""Prints the number of new records in the last 5minutes, hour 24
hours, 3 days, and week

"""

import os
import sys
import datetime
import pytz
import pandas as pd


def dateparse(str):
    return pd.datetime.strptime(str, '%Y-%m-%d-%H-%M-%S')


def main():
    tz_utc = pytz.timezone("UTC")
    now = datetime.datetime.now(tz=tz_utc)

    filename = 'attendance.csv'  # 'abcdefgh'
    filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        filename
    )

    try:
        with open(filepath, 'r') as fh:
            linelist = fh.readlines()
            print(
                "File {0} contains {1} records".format(
                    filename,
                    len(linelist) - 1
                )
            )

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    df = pd.read_csv(
        filepath,
        sep=',',
        parse_dates=['timestamp'],
        date_parser=dateparse
    )

    df['timestamp'] = df['timestamp'].apply(lambda dt: dt.tz_localize(tz_utc))

    COL_WIDTH = 20
    print("\nNumber of records in the last:")

    nb = df[now < df['timestamp'] + pd.Timedelta('7D')].shape[0]
    print("{0}: {1}".format("week".ljust(COL_WIDTH), nb))

    nb = df[now < df['timestamp'] + pd.Timedelta('3D')].shape[0]
    print("{0}: {1}".format("3 days".ljust(COL_WIDTH), nb))

    nb = df[now < df['timestamp'] + pd.Timedelta('24H')].shape[0]
    print("{0}: {1}".format("24 hours".ljust(COL_WIDTH), nb))

    nb = df[now < df['timestamp'] + pd.Timedelta('60m')].shape[0]
    print("{0}: {1}".format("60 minutes".ljust(COL_WIDTH), nb))

    nb = df[now < df['timestamp'] + pd.Timedelta('5m')].shape[0]
    print("{0}: {1}".format("5 minutes".ljust(COL_WIDTH), nb))


if __name__ == '__main__':
    main()
