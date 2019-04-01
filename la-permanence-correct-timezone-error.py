#!/usr/bin/env python
"""WARNING!  THIS SHOULD BE RUN ONLY ONCE - and it
has already been run!

Set timestamp one hour back in rows of dataset attendance.csv recorded
with wrong timezone.

The rows to be corrected correspond to data collected after time change of 31 Mar
2019 (daylight saving time in the UK) until the scraping script has
been corrected.  The latter is detected by finding the row with a
timestamp prior to the timestamp of the previous row.
"""
# Imports
import os
import shutil
import datetime
import pytz
import numpy as np
import pandas as pd

# Constants

TZ_UTC = pytz.timezone("UTC")


filename = 'attendance.csv'
foldername = 'Projects/la_permanence'
filepath = os.path.join(os.path.expanduser("~"), foldername, filename)
filesize = os.path.getsize(filepath)


# Functions
def dateparse(str):
    return pd.datetime.strptime(str, '%Y-%m-%d-%H-%M-%S')


def main():
    print("="*60)
    print("="*60)
    print("WARNING: this should be run only once!")
    print("="*60)
    print("="*60)
    print(
        "This script will modify data in file {0}".format(filename)
    )
    print("to correct for an error in timezone manipulation")
    print("while scraping the data.")

    print("This script will abort now.")
    return

    df = pd.read_csv(
        filepath,
        sep=',',
        dtype={'Moulin': np.uint8, 'Alésia': np.uint8},
        parse_dates=['timestamp'],
        date_parser=dateparse
    )

    df['timestamp'] = \
        df['timestamp'].apply(lambda ts: ts.tz_localize(TZ_UTC))
    df['shifted'] = df['timestamp'].shift(1)

    # Index of first row after time change
    ts0 = pd.Timestamp(2019, 3, 31, 2, 0).tz_localize(TZ_UTC)
    idx0 = df[df['timestamp'] < ts0].index[-1]

    # Index of first row where error is corrected
    idx1 = df[df['timestamp'] < df['shifted']].index[0]

    backup_filename = \
        ".".join([
            "-".join([
                ".".join(filename.split('.')[0:-1]),
                "BACKUP",
                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            ]),
            filename.split('.')[-1]
        ])

    print(
        "Rows between index {0} and {1} (inclusive) will be modifed.".format(
            idx0 + 1,
            idx1
        )
    )
    msg = "Continue (copy of data file {0} will be saved in {1})?".format(
        filename,
        backup_filename
    )
    print(msg)
    answer = input("Yes/[no]\n")

    if not (answer == 'Yes'):
        return

    shutil.copy(filename, backup_filename)
    print("Backup completely.")

    def correct_for_timezone(row):
        idx = row.name
        ts = row['timestamp']

        string = ""
        if (idx <= idx0) or (idx >= idx1):
            string = ts.strftime('%Y-%m-%d-%H-%M-%S')
        else:
            string = (ts - pd.Timedelta('1H')).strftime('%Y-%m-%d-%H-%M-%S')

        return string

    print("Note: region where data is to be modified:")
    print(df.iloc[idx0-2:idx0+2][['timestamp', 'Moulin', 'Alésia']])
    print(df.iloc[idx1-2:idx1+2][['timestamp', 'Moulin', 'Alésia']])


    print("Correct for timezone errors...")
    df['new_ts'] = df.apply(correct_for_timezone, axis=1)
    print("Done.")

    df.drop(['timestamp', 'shifted'], axis=1, inplace=True)
    df.rename(columns={'new_ts': 'timestamp'}, inplace=True)
    df = df[['timestamp', 'Moulin', 'Alésia']]

    print(df.iloc[idx0-2:idx0+2])
    print(df.iloc[idx1-2:idx1+2])
    new_filename = "attendance.csv"  # "attendance-tmp.csv"
    df.to_csv(
        new_filename,
        sep=',',
        index=False
    )
    



if __name__ == "__main__":
    main()
