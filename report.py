#!/usr/bin/env python
"""Prints the number of new records in the last 5 minutes, hour 24
hours, 3 days, and week

"""
# Imports
import os
import sys
import datetime
import pytz
import math
import pandas as pd

# Constants
BLACK = "30"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
MAGENTA = "35"
CYAN = "36"
WHITE = "37"

BGBLACK = "40"
BGRED = "41"
BGGREEN = "42"
BGYELLOW = "43"
BGBLUE = "44"
BGMAGENTA = "45"
BGCYAN = "46"
BGWHITE = "47"

TIMESTAMP_WIDTH = 15
ELAPSED_WIDTH = 15


# Functions
def dateparse(str):
    return pd.datetime.strptime(str, '%Y-%m-%d-%H-%M-%S')


def elapsed_strftime(td):
    hours = td.seconds // (60*60)
    mins = td.seconds // 60 - 60 * hours
    string = "{0:2d}h {1:2d}m".format(hours, mins)
    return string


def row_strf(row):
    timestamp = row['timestamp'].strftime('%H:%M:%S').ljust(TIMESTAMP_WIDTH)
    timestamp = color_text(timestamp, color=CYAN)
    elapsed = elapsed_strftime(row['elapsed']).ljust(ELAPSED_WIDTH)
    cumsum = row['cumsum']
    count = row['count']

    if count == 0:
        color = RED
    else:
        color = GREEN

    text = \
        "{0} ({1})".format(
            cumsum, color_text("+{}".format(count), color=color)
        )

    numerics = \
        "   ".join([
            color_text("*"*(math.ceil(cumsum / 5)), color=color),
            text
        ])

    return "".join([
        timestamp,
        elapsed,
        numerics
    ])


def color_format(color=WHITE):
    CIN, COUT = "\x1B[", "m"
    return CIN + color + COUT


def color_text(txt, color=WHITE):
    return "".join([
        color_format(color),
        txt,
        '\x1B[0m'  # reset color
    ])


def main():
    tz_utc = pytz.timezone("UTC")
    now = datetime.datetime.now(tz=tz_utc)

    filename = 'attendance.csv'
    filepath = os.path.join(
        os.path.expanduser("~"),
        'Projects/la_permanence',
        filename
    )

    print("Report on recent data collection".upper())
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

    raw_data = pd.read_csv(
        filepath,
        sep=',',
        parse_dates=['timestamp'],
        date_parser=dateparse
    )
    raw_data['timestamp'] = \
        raw_data['timestamp'].apply(lambda dt: dt.tz_localize(tz_utc))

    ts_start = now - pd.Timedelta('7days 1hours')
    ts_start = pd.Timestamp(
        ts_start.year,
        ts_start.month,
        ts_start.day,
        ts_start.hour,
        5*(ts_start.minute // 5)
    ).tz_localize(tz_utc)

    timestamps = pd.DataFrame(
        data=pd.date_range(start=ts_start, end=now, freq='5min'),
        columns=['timestamp']
        )

    df = raw_data[raw_data['timestamp'] >= ts_start].copy()

    df['5 minutes'] = \
        df['timestamp'].apply(
            lambda dt: pd.Timestamp(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                5 * (dt.minute // 5)
            ).tz_localize(tz_utc)
        )

    dg = \
        df.groupby('5 minutes')['timestamp'].count().reset_index().rename(
            columns={'timestamp': 'count', '5 minutes': 'timestamp'}
        )

    dg = pd.merge(
        left=timestamps,
        right=dg,
        on='timestamp',
        how='left'
    ).fillna(value=0)
    dg['count'] = dg['count'].astype(int)

    dg['elapsed'] = now - dg['timestamp']

    dg['cumsum'] = \
        dg[['count']].sort_index(ascending=False).\
        cumsum(axis=0).sort_index(ascending=True)

    dg = dg[['timestamp', 'elapsed', 'cumsum', 'count']]

    rows = [row_strf(row) for index, row in dg.iterrows()]

    top_bar = "".join([
        color_text("timestamp", color=CYAN).ljust(TIMESTAMP_WIDTH),
        "elapsed".ljust(ELAPSED_WIDTH),
        "reverse cumulative (+added in 5-minute block)"
    ])

    top_bar = "".join([
        "timestamp".ljust(TIMESTAMP_WIDTH),
        "elapsed".ljust(ELAPSED_WIDTH),
        "reverse cumulative (+ added in 5-minute block)"
    ])
    graph = "\n".join(rows[-20:])

    print("-"*80)
    print(top_bar)
    print("-"*80)
    print(graph)

    COL_WIDTH = 20
    print("\nSummary of recent activity: number of records in the last\n")

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
