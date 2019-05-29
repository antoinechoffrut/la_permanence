#!/usr/bin/env python3
"""Prints the number of new records in the last 5 minutes, hour 24
hours, 3 days, and week

"""
# Imports
import os
# import sys
import datetime
import pytz
import math
import pandas as pd

# Constants
TZ_UTC = pytz.timezone("UTC")

LOCAL_DIR = os.path.join(
    os.path.expanduser('~'),
    'Data',
    'la_permanence'
)
FILENAME = 'availability.csv'  # formerly: 'attendance.csv'
FILEPATH = os.path.join(LOCAL_DIR, FILENAME)

SECONDS_IN_ONE_MINUTE = 60
MINUTES_IN_ONE_HOUR = 60
MINUTES_IN_ONE_DAY = 24*MINUTES_IN_ONE_HOUR
MINUTES_IN_ONE_WEEK = 7*MINUTES_IN_ONE_DAY

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

TIMESTAMP_WIDTH = 20
ELAPSED_WIDTH = 20


# Functions
def dateparse(str):
    return pd.datetime.strptime(str, '%Y-%m-%d %H:%M:%S')
    # return pd.datetime.strptime(str, '%Y-%m-%d-%H-%M-%S')


def elapsed_strftime(td):
    hours = td.seconds // (SECONDS_IN_ONE_MINUTE*MINUTES_IN_ONE_HOUR)
    mins = td.seconds // SECONDS_IN_ONE_MINUTE - MINUTES_IN_ONE_HOUR * hours
    string = "{0:2d}h {1:2d}m".format(hours, mins)
    return string


def row_strf(row):
    timestamp = row['timestamp'].strftime('%H:%M:%S').ljust(TIMESTAMP_WIDTH)
    timestamp = color_text(timestamp, color=CYAN)
    elapsed = elapsed_strftime(row['elapsed']).ljust(ELAPSED_WIDTH)
    cumsum = row['cumsum']
    count = row['count']

    if count == 5:
        color = GREEN
    elif count == 0:
        color = RED
    else:
        color = YELLOW

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


def does_file_exist(filepath=FILEPATH):
    filename = filepath.split('/')[-1]
    try:
        with open(filepath, 'r') as fh:
            linelist = fh.readlines()
            print(
                "File {0} contains {1} records".format(
                    filename,
                    len(linelist) - 1
                )
            )
        return True
    except FileNotFoundError as e:
        print(e)
        return False


def load_data(filepath, ts_start):
    # tz_utc = pytz.timezone("UTC")
    df = pd.read_csv(
        filepath,
        sep=',',
        # parse_dates=['timestamp'],
        # date_parser=dateparse
    )
    # First rough filter, keep enough rows to cover last week
    df = df.iloc[-MINUTES_IN_ONE_WEEK:]
    df['timestamp'] = df['timestamp'].apply(dateparse)
    df['timestamp'] = \
        df['timestamp'].apply(lambda dt: dt.tz_localize(TZ_UTC))

    df = df[df['timestamp'] >= ts_start]

    return df


def get_counts(df, now, timestamps):
    """Return counts of timestamps within every 5-minute intervals over
    the past week (plus one hour)"""

    dg = df.copy()

    dg['5 minutes'] = \
        dg['timestamp'].apply(
            lambda dt: pd.Timestamp(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                5 * (dt.minute // 5)
            ).tz_localize(TZ_UTC)
        )

    dg = \
        dg.groupby('5 minutes')['timestamp'].count().reset_index().rename(
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

    return dg


def get_5min_timestamps():
    """Return timestamps for
    - current date and time;
    - date and time one week (plus one hour) ago; and
    - dataframe with timestamps spaced every 5 minutes in between.
    """
    now = datetime.datetime.now(tz=TZ_UTC)
    ts_start = now - pd.Timedelta('7days 1hours')
    ts_start = pd.Timestamp(
        ts_start.year,
        ts_start.month,
        ts_start.day,
        ts_start.hour,
        5*(ts_start.minute // 5)
    ).tz_localize(TZ_UTC)

    timestamps = pd.DataFrame(
        data=pd.date_range(start=ts_start, end=now, freq='5min'),
        columns=['timestamp']
    )

    return now, ts_start, timestamps


def display_info(df, dg, now):
    """Print to terminal timestamp count info from dataframe."""
    rows = [row_strf(row) for index, row in dg.iterrows()]

    # top_bar = "".join([
    #     color_text("timestamp", color=CYAN).ljust(TIMESTAMP_WIDTH),
    #     "elapsed".ljust(ELAPSED_WIDTH),
    #     "reverse cumulative (+added in 5-minute block)"
    # ])

    top_bar = "".join([
        "timestamp (UTC)".ljust(TIMESTAMP_WIDTH),
        "elapsed since".ljust(ELAPSED_WIDTH),
        "reverse cumulative (+ added in 5-minute block)"
    ])
    graph = "\n".join(rows[-20:])

    print("-"*80)
    print(top_bar)
    print("-"*80)
    print(graph)

    print("\nLast data collected:")
    print(df.iloc[-3:])

    print("\nSummary of recent activity: number of records in the last\n")

    for (freq, label) in [
            ('7D', '1 week'),
            ('3D', '3 days'),
            ('24H', '24 hours'),
            ('60m', '60 minutes'),
            ('5m', '5 minutes')
    ]:

        string = fmt_summary(df, now, freq=freq, label=label)
        print(string)

    return


def fmt_summary(df, now, freq, label):
    LABEL_WIDTH = 20

    label = label.ljust(LABEL_WIDTH)

    max_hits = \
        pd.Timedelta(freq).days * MINUTES_IN_ONE_DAY \
        + pd.Timedelta(freq).seconds // SECONDS_IN_ONE_MINUTE

    count = df[now < df['timestamp'] + pd.Timedelta(freq)].shape[0]

    percentage = f'{count/max_hits:.0%}'
    percentage = f'({percentage})'.rjust(6)

    count = str(count).rjust(6)

    string = "{0}: {1}  {2}".format(
        label,
        count,
        percentage
    )
    return string


def main():
    print('-'*60)
    print("La Permanence - recently collected data report")

    if not does_file_exist(filepath=FILEPATH):
        return

    now, ts_start, timestamps = get_5min_timestamps()

    df = load_data(filepath=FILEPATH, ts_start=ts_start)

    dg = get_counts(df, now, timestamps)

    display_info(df, dg, now)
    fmt_ts = '%Y-%m-%d %H:%M:%S'
    tz_utc = pytz.timezone("UTC")
    run_time = datetime.datetime.now(tz=tz_utc)
    timestamp = run_time.strftime(fmt_ts)
    print(f"\nCurrent timestamp (UTC): {timestamp}\n")


if __name__ == '__main__':
    main()
