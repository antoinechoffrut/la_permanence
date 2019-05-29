#!/usr/bin/env python
"""
New version of main module for La Permanence project.

There are three main data files:
- local: the file containing the data collected by this machine
- kaggle: the file saved on kaggle
- combined: the file obtained by merging the local and kaggle files

"""

# Imports
import os
import re
import zipfile
import tempfile
import pytz
import datetime
import shlex
import subprocess
import logging
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Constants
FILENAME = 'availability.csv'
# FILENAME = 'abc.csv'            # for debugging purposes

LOCAL_DIR = os.path.join(
    os.path.expanduser('~'), 'Data/la_permanence/'
)
COMBINED_DIR = os.path.join(
    os.path.expanduser('~'), 'kaggle/la_permanence/'
)
SCRIPT_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/scripts/'
)
# TMP_DIR = os.path.join(
#     os.path.expanduser('~'), 'Projects/la_permanence/tmp/'
# )
TMP_DIR = tempfile.gettempdir()
BACKUP_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/DATA_BACKUPS/'
)
WEB_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/la-permanence-web/'
)
FIG_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/la-permanence-web/_images'
)


TZ_UTC = pytz.timezone("UTC")
TZ_PARIS = pytz.timezone("Europe/Paris")

ONE_HOUR = 60
HOURLY = [h * ONE_HOUR for h in range(0, 24)]

ONE_DAY = 24*ONE_HOUR
DAILY = [d * ONE_DAY for d in range(0, 7)]

ONE_WEEK = 7*ONE_DAY
WEEKLY = [w * ONE_WEEK for w in range(0, 52)]

DAYS_OF_THE_WEEK = {0: 'lundi',
                    1: 'mardi',
                    2: 'mercredi',
                    3: 'jeudi',
                    4: 'vendredi',
                    5: 'samedi',
                    6: 'dimanche'}


# Functions
# Helper functions

def bytes_to_msg(result):
    """Convert and format bytes to list of strings.

    This is used to format output and error messages from stdout and stderr.
    """
    msg = result.decode()
    msg = re.sub(r'\n$', '', msg)
    msg = msg.split('\n')
    msg = [line.split('\r')[-1] for line in msg]
    return msg


def subprocess_command(command, cwd=None):
    """Run command and return exit code, stdout and stderr.

    The parameter command is the string that would be passed at the
    command prompt.
    If the parameter cwd is not None, the function changes the working
    directory to cwd.  (This is used when pushing to github.)
    """
    args = shlex.split(command, posix=False)

    p = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
        )
    try:
        outs, errs = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        outs, errs = p.communicate()
    finally:
        returncode = p.returncode

    return returncode, outs, errs


def log_messages(msg, returncode):
    """Log messages.

    The parameter msg, if not None, is a list of strings of
    characters.  Each string contains no newline nor carriage return,
    and therefore represents a line to print to logfile.

    The parameter returncode is the exit status of the command whose
    message this function is logging.  The message is printed as info
    if returncode is 0 and as error otherwise.

    """
    if (not msg) or (msg == ['']):
        return

    logger = logging.getLogger()

    if returncode == 0:
        for line in msg:
            logger.info(line)
    else:
        for line in msg:
            logger.error(line)


# Decorators
def log(func):
    def wrapper():
        returncode, outs, errs = func()
        out_msg = bytes_to_msg(outs)
        log_messages(out_msg, returncode=returncode)
        return returncode, outs, errs
    return wrapper


# Commands
@log
def download_from_kaggle():
    command = ' '.join([
        'kaggle datasets download',
        f'-f {FILENAME}',
        f'-p {TMP_DIR}',
        '--force',
        'antoinechoffrut/la-permanence'
    ])
    returncode, outs, errs = subprocess_command(command)

    return returncode, outs, errs


@log
def upload_to_kaggle():
    """Upload dataset to kaggle.

    The dataset is in the file ~/kaggle/la_permanence/availability.csv
    and contains the data collected locally on this computer with the
    previous version of the data from kaggle.
    """
    unwanted_file = os.path.join(COMBINED_DIR, '.DS_Store')
    if os.path.exists(unwanted_file):
        os.remove(unwanted_file)

    command = ' '.join([
        'kaggle datasets version',
        '-d',
        # f'-p {COMBINED_DIR}',
        '-p bob',
        '-m Update'
    ])

    returncode, outs, errs = subprocess_command(command)

    return returncode, outs, errs


@log
def git_add_figures():
    cwd = FIG_DIR
    fignames = []
    """Stage figures moulin-summary.png and alesia-summary.png."""
    for location in ['moulin', 'alesia']:
        fignames.append(
            os.path.join(
                FIG_DIR,
                '-'.join([location, 'summary']) + '.png'
            )
        )
    # fignames = ['bob.png']  # for debugging purposes

    command = f"git add {' '.join(fignames)}"
    returncode, outs, errs = \
        subprocess_command(command, cwd=cwd)

    return returncode, outs, errs


@log
def git_commit_figures():
    """Commit staged figures."""
    cwd = WEB_DIR

    command = 'git commit -m "Update figures"'

    returncode, outs, errs = \
        subprocess_command(command, cwd=cwd)

    return returncode, outs, errs


@log
def git_push_figures():
    """Push figures"""
    cwd = WEB_DIR

    command = 'git push origin gh-pages'

    returncode, outs, errs = \
        subprocess_command(command, cwd=cwd)

    return returncode, outs, errs


# Scraping
def get_LaPermanence_page():
    """Retrieve page http://www.la-permanence.com"""
    url = "https://www.la-permanence.com"
    # url = "http://bsegerglb.com"
    logger = logging.getLogger()
    try:
        page = requests.get(url)
    except requests.ConnectionError as e:
        page = None
        logger.error(e)
    return page


def extract_data(page):
    """Extract number of seats at Moulin and Alésia coworking spaces
    from html page."""

    FMT_TS = '%Y-%m-%d %H:%M:%S'
    tz_utc = pytz.timezone("UTC")
    run_time = datetime.datetime.now(tz=tz_utc)
    timestamp = run_time.strftime(FMT_TS)

    soup = BeautifulSoup(page.text, "html.parser")
    locations = soup.find_all(
        "div",
        {"class": "seats"}
    )

    moulin_seats = ""
    alesia_seats = ""

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

    row = ','.join([timestamp, moulin_seats, alesia_seats])
    return row  # moulin_seats, alesia_seats


def write_data_to_file(row):
    """Append row with data to datafile."""
    logger = logging.getLogger()

    os.makedirs(LOCAL_DIR, exist_ok=True)
    path = os.path.join(LOCAL_DIR, FILENAME)

    if not os.path.exists(path):
        logger.warning(f'Datafile {path} did not exist, was created.')
        with open(path, 'w') as file:
            file.write('timestamp,moulin,alesia')
            file.write('\n')

    with open(path, 'a+') as file:
        logger.info(f'Write "{row}" to datafile {path}.')
        file.write(row)
        file.write('\n')
    return


def scrape():
    page = get_LaPermanence_page()
    if page is None:
        return
    row = extract_data(page)
    write_data_to_file(row)


def combine_kaggle_local():
    logger = logging.getLogger()
    logger.info("Load local dataset...")
    dm_local = DataManager(source='local')
    logger.info("Load dataset from kaggle...")
    dm_kaggle = DataManager(source='kaggle')
    logger.info("Combine datasets...")
    dm_combined = dm_kaggle.merge(dm_local)
    filepath = os.path.join(COMBINED_DIR, FILENAME)
    logger.info(f"Save combined dataset to {filepath}")
    dm_combined.data.to_csv(filepath)


# Classes
class Location:

    LOC_INFO = {
        'moulin': {
            'name': 'moulin',
            'address': "rue du Fer à Moulin",
            'max_seats': 65,
        },
        'alesia': {
            'name': 'alesia',
            'address': "rue d'Alésia",
            'max_seats': 82
        }
    }

    def __init__(self, name):
        if name not in self.LOC_INFO.keys():
            logger = logging.getLogger()
            logger.error(f'Invalid location name "{name}"')
            return None
        for key in self.LOC_INFO[name].keys():
            setattr(self, key, self.LOC_INFO[name][key])

    def seat_ticks(self):
        ticks = \
            list(range(0, 10 * (2 + self.max_seats // 10), 10)) \
            + [self.max_seats]
        ticklabels = \
            [str(n) for n in ticks[:-1]] + [f'max: {ticks[-1]}']
        z = zip(ticks, ticklabels)
        z = sorted(z, key=lambda x: x[0])
        ticks, ticklabels = zip(*z)
        return ticks, ticklabels


class DataManager:
    TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

    DAY_MODES = {
        'DayOfWeek': 'Day of week',
        'DayOfYear': 'Day of year',
        'Date': 'Date'
    }
    WEEK_MODES = {
        'WeekOfYear': 'Week of year',
        'WeekOf': 'Week of'}
    MODES = {**DAY_MODES, **WEEK_MODES}

    MODE_TO_INDEX = {
        'DayOfWeek': 'MinuteOfDay',
        'DayOfYear': 'MinuteOfDay',
        'Date': 'MinuteOfDay',
        'WeekOfYear': 'MinuteOfWeek',
        'WeekOf': 'MinuteOfWeek'
    }

    def __init__(self, data=None, source='local', nrows=None):
        self.data = self.load_dataset(data, source, nrows)

    @classmethod
    def load_dataset(cls, data=None, source='local', nrows=None):
        logger = logging.getLogger()
        if data is not None:
            return data.copy()

        if source == 'local':
            path = os.path.join(LOCAL_DIR, FILENAME)
        elif source == 'combined':
            path = os.path.join(COMBINED_DIR, FILENAME)
        elif source == 'kaggle':
            download_from_kaggle()
            path = os.path.join(TMP_DIR, FILENAME)
            zip_file = path + '.zip'
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(TMP_DIR)
            os.remove(zip_file)
        else:
            return None

        if not os.path.exists(path):
            logger.error(f"No file at {path}.")
            return None

        with open(path, 'r') as file:
            first_line = file.readline()

        columns = first_line.split('\n')[0].split(',')
        valid_columns = \
            ('timestamp' in columns) and \
            (('moulin' in columns) or ('alesia' in columns))
        if not valid_columns:
            logger.error(f"Incorrect column names: {columns}.")
            return None
        data = pd.read_csv(
            path,
            parse_dates=['timestamp'],
            date_parser=cls.dateparse,
            nrows=nrows)

        data['timestamp'] = \
            data['timestamp'].apply(
                lambda dt: dt.tz_localize(TZ_UTC).tz_convert(TZ_PARIS)
            )

        data.set_index("timestamp", inplace=True)

        return data

    @staticmethod
    def dateparse(timestamp, timestamp_format=TIMESTAMP_FORMAT):
        return pd.datetime.strptime(timestamp, timestamp_format)

    def resample(self, resol=5):
        self.data = \
            self.data.resample(
                f'{resol}T'
            ).mean().interpolate().round().astype(np.uint8)

    def augment_features(self):
        data = self.data

        data.reset_index(inplace=True)

        data['Date'] = data['timestamp'].apply(
            lambda ts: pd.Timestamp(
                ts.year, ts.month, ts.day
            ).tz_localize(TZ_PARIS)
        )

        attributes = \
            ['Minute', 'Hour', 'Day', 'DayOfWeek', 'WeekOfYear', 'DayOfYear']
        for attr in attributes:
            data[attr] = getattr(data['timestamp'].dt, attr.lower())

        data['WeekOf'] = data['Date'] - data['DayOfWeek']*pd.Timedelta(days=1)

        data['MinuteOfDay'] = 60*data['Hour'] + data['Minute']

        data['MinuteOfWeek'] = 24*60*data['DayOfWeek'] + data['MinuteOfDay']

        for col in ['Date', 'WeekOf']:
            data[col] = pd.to_datetime(data[col].dt.date)
        data.set_index('timestamp', inplace=True)

    def create_view(self, location, mode):

        index = DataManager.MODE_TO_INDEX[mode]
        df = self.data.loc[:, [location.name, mode, index]].copy()
        if df.index.freq is None:
            logger = logging.getLogger()
            logger.warning("Must resample data before creating view.")
            return None

        df.rename(columns={location.name: 'availability'}, inplace=True)
        table = pd.pivot_table(
            df,
            index=index,
            columns=mode,
            values='availability'
        )

        return View(table, location)

    def past_week(self):
        """Generate DataManager object from data of past week."""
        last_timestamp = self.data.index.max()
        last_week_timestamp = last_timestamp - pd.Timedelta('7D')
        last_week = self.data[self.data.index >= last_week_timestamp].copy()

        return DataManager(last_week)

    def merge(self, other):
        df_left = self.data.copy()
        df_right = other.data.copy()
        for dg in [df_left, df_right]:
            dg.reset_index(inplace=True)
            dg['timestamp'] = \
                dg['timestamp'].apply(lambda ts: ts.replace(second=0))

        df = pd.merge(
            left=df_left,
            right=df_right,
            on=['timestamp', 'moulin', 'alesia'],
            how='outer',
            # indicator=True
        )
        df.drop_duplicates(subset='timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)

        return DataManager(data=df)


class View:
    def __init__(self, table, location):
        self.table = table
        self.index = self.table.index.name
        # self.mode = self.table.columns.name
        # self.freq = freq
        self.location = location

    def split_columns(self):
        tables = [
            self.table[col].rename(col.strftime('%Y-%m-%d')).to_frame()
            for col in self.table.columns
        ]

        for table in tables:
            table.columns.name = self.table.columns.name

        views = [View(table=table, location=self.location)
                 for table in tables]
        return views

    def mean(self):
        table = self.table.mean(axis=1).to_frame(name='mean')
        table.columns.name = self.table.columns.name
        location = self.location
        return View(table, location)

    def median(self):
        table = self.table.median(axis=1).to_frame(name='median')
        table.columns.name = self.table.columns.name
        location = self.location
        return View(table, location)

    def std(self):
        table = self.table.std(axis=1).to_frame(name='std')
        table.columns.name = self.table.columns.name
        location = self.location
        return View(table, location)

    def min(self):
        table = self.table.min(axis=1).to_frame(name='min')
        table.columns.name = self.table.columns.name
        location = self.location
        return View(table, location)

    def max(self):
        table = self.table.max(axis=1).to_frame(name='max')
        table.columns.name = self.table.columns.name
        location = self.location
        return View(table, location)

    def clip(self, lower=None, upper=None, *args, **kwargs):
        location = self.location
        if ('inplace' in kwargs.keys()) and (kwargs['inplace'] is True):
            self.table.clip(lower=lower, upper=upper, axis=1, *args, **kwargs)
            return None
        else:
            table = self.table.clip(
                lower=lower, upper=upper,
                axis=1, *args, **kwargs)
            return View(table, location)

    def compatible_with(self, other):
        res = \
            (self.index == other.index) \
            and (self.table.columns.name == other.table.columns.name) \
            and (self.table.index.equals(other.table.index)) \
            and (self.location == other.location)
        return res

    def __add__(self, other):
        logger = logging.getLogger()
        if not self.compatible_with(other):
            logger.error("Views not compatible")
            return None
        values = self.table.values + other.table.values
        table = pd.DataFrame(data=values, index=self.table.index)
        location = self.location
        return View(table, location)

    def __sub__(self, other):
        logger = logging.getLogger()
        if not self.compatible_with(other):
            logger.error("Views not compatible")
            return None
        values = self.table.values - other.table.values
        table = pd.DataFrame(data=values, index=self.table.index)
        location = self.location
        return View(table, location)

    def __rmul__(self, scalar):
        return View(scalar*self.table, self.location)

    def color_cycle(self):
        if self.index == 'MinuteOfDay':
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][0:7]
        if self.index == 'MinuteOfWeek':
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][0:1]

    def legend_labels(self):
        if self.index == 'MinuteOfDay':
            return DAYS_OF_THE_WEEK.values()
        if self.index == 'MinuteOfWeek':
            return []

    def row_label(self):
        if self.index == 'MinuteOfDay':
            return 'Time of day'
        if self.index == 'MinuteOfWeek':
            return 'Day of week'

    def row_ticks(self, plot_type):
        if self.index == 'MinuteOfDay':
            ticks = HOURLY
            ticklabels = \
                ["{0:02d}h00".format(x)
                 for x in range(0, 24)]

        elif self.index == 'MinuteOfWeek':
            ticks = [(h*ONE_HOUR + d*ONE_DAY)
                     for h in range(0, 24, 6) for d in range(0, 7)]
            ticks = ticks + [ONE_WEEK]
            ticklabels = \
                [DAYS_OF_THE_WEEK[int(x // ONE_DAY)].ljust(12)
                 if (x % ONE_DAY == 0) and (x < ONE_WEEK)
                 else f'{(x % ONE_DAY) // ONE_HOUR:02d}h'
                 for x in ticks]

        if (plot_type == 'heatmap'):
            ticks = list(self.table.index)

        return ticks, ticklabels

    def col_label(self):
        return DataManager.MODES[self.table.columns.name]

    def col_ticks(self, plot_type):

        if self.table.columns.name == 'Date':
            date_range = \
                pd.date_range(
                    self.table.columns.min(),
                    self.table.columns.max(),
                    freq='D')
            daysofweek = [dt.dayofweek for dt in date_range]
            dates = \
                [dt.strftime('%Y-%m-%d') for dt in date_range]

            enum = enumerate(zip(daysofweek, dates))

            ticks, ticklabels = \
                zip(*[
                    (idx + 0.5, date)
                    for (idx, (dayofweek, date)) in enum
                    if dayofweek == 0
                ])
            return ticks, ticklabels

        if self.table.columns.name == 'DayOfYear':
            vals = self.table.columns.values
            enum = enumerate(vals)
            ticks, ticklabels = \
                zip(*[
                    (idx + 0.5, val)
                    for idx, val in enum
                    if idx % 7 == 0
                ])

            return ticks, ticklabels

        if self.table.columns.name == 'DayOfWeek':
            vals = [the_day[0:3]
                    for the_day in DAYS_OF_THE_WEEK.values()]
        elif self.table.columns.name == 'WeekOf':
            date_min = self.table.columns.min()
            date_max = self.table.columns.max()
            date_range = \
                pd.date_range(date_min, date_max, freq='W')
            vals = [dt.strftime('Week of %Y-%m-%d')
                    for dt in self.table.columns]
        elif self.table.columns.name == 'WeekOfYear':
            vals = self.table.columns.values

        enum = enumerate(vals)
        ticks, ticklabels = \
            zip(*[
                (idx + 0.5, val)
                for idx, val in enum
            ])

        return ticks, ticklabels


# main
def joe():
    moulin = Location('moulin')
    # alesia = Location('alesia')
    dm = DataManager(source='local', nrows=20000)
    # dm.resample()
    # print(dm.data.info())
    dm.augment_features()
    print(dm.data.index.freq)
    # print(dm.data.info())
    view = dm.create_view(location=moulin, mode='WeekOf')
    print(view.table.info())
    print(view.table.tail())

    view_mean = view.mean()
    print(view_mean.table.head())


def main():
    script_name = os.path.basename(__file__)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_log = os.path.join(
        script_dir,
        re.sub(".py$", ".log", script_name)
    )
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d | %(levelname)-10s: %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(script_log, mode='w'),
            logging.StreamHandler()
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.info("Combine kaggle and local datasets:")
    combine_kaggle_local()


if __name__ == '__main__':
    main()
