#!/usr/bin/env python
"""Main module for La Permanence project.

This project consists in collecting the number of seats available at
the coworking space La Permanence in Paris.

This data is summarized and published on a github page at:
https://antoinechoffrut.github.io/la-permanence-web/

There are two locations:
- "Moulin" in rue du Fer à Moulin
- "Alésia" in rue d'Alésia

There are three main data files.  They are all named availability.csv
but are saved in different directories.

- local: the file containing the data collected by this machine,
located in ~/Data/la_permanence

- kaggle: the file saved on
https://www.kaggle.com/antoinechoffrut/la-permanence

- combined: the file obtained by merging the local and kaggle
datasets, located in ~/kaggle/la_permanence

The reason for this is that data is collected on separate machines.
On one machine, not used for any other purpose, the scraping is done
every minute, but occasionally the connection breaks or the computer
shuts down unexpectedly.  It has happened that several days or even
weeks have been missed.  The other machine is a laptop for daily use,
and therefore the lid is frequently closed, preventing the scraping.
Each computer retains its own copy of the data it has collected, and
each generates another dataset by combining its own with that from
kaggle.

The datafiles are .csv files with three fields:
- timestamp: in UTC with "naive" formatting;
- moulin: number of available seats at the Moulin location;
- alesia: number of available seats at the Alésia location.

The fields are separated by commas.  For example, the first row in the
file is

2019-01-08 14:49:06,4,9

meaning that on 8 Jan 2019 at 14h49m06s (UTC) there were 4 and 9
available seats at the Moulin and Alésia locations respectively

The "naive" formatting of the timestamp means that "+00:00" is
omitted.

The class DataManager is a convenience class for loading the data from
the three sources: "local", "kaggle", "combined".  Its main (and
currently only) attribute `data` is a pandas.DataFrame containing the
data from one of these source datafiles.  The index of `data` contains the
values from the column `timestamp` in the .csv file, and upon
instantiation `data` contains two columns `moulin` and `alesia` with
the data from the columns with same names in the datafile.

The index `timestamp` can be of one of two types only:
- naive (timezone-unaware), in which case it represents the date and
time in UTC;
- timezone-aware, in the local Paris timezone.

Upon instantation, the index is of naive datetime type.

There is a method to localize and convert to the Paris timezone, and
method to revert to the naive UTC type.

Care must be taken when combining datasets and writing the combined
data to file.

Figures are generated summarizing the history of the data.  These
figures are saved in
~/Projects/la_permanence/la-permanence-web/assets and are used on the
github page.

The repository for the project is at
https://github.com/antoinechoffrut/la_permanence

The repository for this github page is at
https://github.com/antoinechoffrut/la-permanence-web

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

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# import seaborn as sns


# Constants

# Common name for all datafiles
FILENAME = 'availability.csv'
# FILENAME = 'abc.csv'            # for debugging purposes

# Directories containing datafiles
LOCAL_DIR = os.path.join(
    os.path.expanduser('~'), 'Data/la_permanence/'
)
COMBINED_DIR = os.path.join(
    os.path.expanduser('~'), 'kaggle/la_permanence/'
)

# Directory where backups of local dataset are saved
BACKUP_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/DATA_BACKUPS/'
)
# Directory to store downloaded datafile from kaggle
TMP_DIR = tempfile.gettempdir()

# Directory of all scripts for La Permanence project
SCRIPT_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/scripts/'
)

# Directory of github page
WEB_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/la-permanence-web/'
)

# Subdirectory for images of github page
FIG_DIR = os.path.join(
    os.path.expanduser('~'), 'Projects/la_permanence/la-permanence-web/assets'
)

# Format of timestamp in datafiles
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

# Timezone constants
DTYPE_NONE = np.dtype('datetime64[ns]')
TZ_UTC = pytz.timezone("UTC")
TZ_PARIS = pytz.timezone("Europe/Paris")
DTYPE_PARIS = \
    pd.core.dtypes.dtypes.DatetimeTZDtype.construct_from_string(
        'datetime64[ns, Europe/Paris]'
    )

# Date and time constants
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

# Constant for plotting
TITLE_FONTSIZE = 32
YLABEL_FONTSIZE = 24


# Functions
# Helper functions

def outputs_to_msg(result):
    """Convert and format bytes to list of strings.

    This is used to format outputs from subprocess.Popen, which
    returns objects of type bytes.
    """
    # msg = result.decode()
    msg = result
    msg = re.sub(r'\r\n', r'\n', msg)
    msg = re.sub(r'\n{2,}', r'\n', msg)
    msg = re.sub(r'\n$', '', msg)
    msg = re.sub(r'^\n', '', msg)
    msg = msg.split('\n')
    msg = [line.split('\r')[-1] for line in msg]

    return msg


def subprocess_command(command, cwd=None, timeout=60, universal_newlines=True):
    """Run command and return exit code, stdout and stderr.

    The parameter command is the string that would be passed at the
    command prompt.
    If the parameter cwd is not None, the function changes the working
    directory to cwd.  (This is used when pushing to github.)

    The function returns:
    - returncode: exit status of command
    - outs: stdout from subprocess
    - errs: stderr from subprocess

    Note:  currently both standard output and standard error streamed
    are combined into one, so that errs is empty.
    """
    logger = logging.getLogger()
    outs = ''
    errs = ''
    args = shlex.split(command, posix=False)

    p = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
        )
    try:
        out, err = p.communicate(timeout=timeout)
        outs = '\n'.join([outs, out.decode()]) if out is not None else outs
        errs = '\n'.join([errs, err.decode()]) if err is not None else errs

    except subprocess.TimeoutExpired as e:
        logger.debug("TimeoutExpired")
        outs = '\n'.join([outs, str(e)]) if e is not None else outs

        p.kill()

        out, err = p.communicate()
        outs = '\n'.join([outs, out.decode()]) if out is not None else outs
        errs = '\n'.join([errs, err.decode()]) if err is not None else errs

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
    """Decorator function to log outputs of subprocess command."""
    def wrapper():
        returncode, outs, errs = func()
        out_msg = outputs_to_msg(outs)

        log_messages(out_msg, returncode=returncode)

        return returncode, outs, errs

    return wrapper


# Commands
@log
def download_from_kaggle():
    """Download dataset availability.csv from
    https://www.kaggle.com/antoinechoffrut/la-permanence and save it
    to Python's tempfile temporary directory.

    Returns: returncode (exit status), outs (stdout), errs (stderr)

    """
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
    """Upload combined dataset to kaggle.

    The combined dataset is in the file
    ~/kaggle/la_permanence/availability.csv and is uploaded to
    https://www.kaggle.com/antoinechoffrut/la-permanence.

    Returns: returncode (exit status), outs (stdout), errs (stderr)

    """
    unwanted_file = os.path.join(COMBINED_DIR, '.DS_Store')
    if os.path.exists(unwanted_file):
        os.remove(unwanted_file)

    command = ' '.join([
        'kaggle datasets version',
        '-d',
        f'-p {COMBINED_DIR}',
        # '-p bob',  # for testing purposes
        '-m Update'
    ])

    returncode, outs, errs = subprocess_command(command)

    return returncode, outs, errs


@log
def git_add_figures():
    """Stage figures moulin-summary.png and alesia-summary.png in github
    page repository.

    The figures are located in
    ~/Projects/la_permanence/la-permanence-web/assets.  The github
    page is at https://antoinechoffrut.github.io/la-permanence-web/.
    The remote repository for the github page is at
    https://github.com/antoinechoffrut/la-permanence-web

    """
    logger = logging.getLogger()
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
    logger.debug(command)
    returncode, outs, errs = \
        subprocess_command(command, cwd=cwd)

    return returncode, outs, errs


@log
def git_commit_figures():
    """Commit staged figures moulin-summary.png and alesia-summary.png in
    github page repository.

    The figures are located in
    ~/Projects/la_permanence/la-permanence-web/assets.  The github
    page is at https://antoinechoffrut.github.io/la-permanence-web/.
    The remote repository for the github page is at
    https://github.com/antoinechoffrut/la-permanence-web

    """

    cwd = WEB_DIR

    command = 'git commit -m "Update figures"'

    returncode, outs, errs = \
        subprocess_command(command, cwd=cwd)

    return returncode, outs, errs


@log
def git_push_figures():
    """Push commit with newly generated figures moulin-summary.png and
    alesia-summary.png in github page repository.

    The figures are located in
    ~/Projects/la_permanence/la-permanence-web/assets.  The github
    page is at https://antoinechoffrut.github.io/la-permanence-web/.
    The remote repository for the github page is at
    https://github.com/antoinechoffrut/la-permanence-web

    """

    cwd = WEB_DIR

    command = 'git push origin gh-pages'

    returncode, outs, errs = \
        subprocess_command(command, cwd=cwd)

    return returncode, outs, errs


# Scraping
def get_LaPermanence_page():
    """Retrieve page http://www.la-permanence.com.

    Returns: page
    """
    url = "https://www.la-permanence.com"
    # url = "http://bsegerglb.com"  # for testing purposes
    logger = logging.getLogger()
    try:
        page = requests.get(url)
    except requests.ConnectionError as e:
        page = None
        logger.error(e)
    return page


def extract_data(page):
    """Extract number of seats at Moulin and Alésia coworking spaces
    from html page.

    Returns: row, a string with timestamp and number of seats at the
    Moulin (x) and Alésia (y) spaces, in the format

    YYYY-MM-DD hh:mm:ss,x,y
    """

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
    marcadet_seats = ""

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
        elif "Marcadet" in location.find_all("p")[0].text:
            marcadet_seats = re.sub(
                "Places",
                "",
                location.find_all("span")[0].text
            ).strip()

    row = ','.join([timestamp, moulin_seats, alesia_seats, marcadet_seats])
    return row  # moulin_seats, alesia_seats, marcadet_seats


def write_data_to_file(row):
    """Append row with data to datafile.

    The row contains the timestamp and number of available seats at
    the Moulin (x) and Alésia (y) coworking spaces in the format

    YYYY-MM-DD hh:mm:ss,x,y

    The datafile availability.csv is located ~/Data/la_permanence/
    """
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
    """Scrape page https://www.la-permanence.com for number of available
    seats at Moulin (x) and Alésia (y) coworking spaces.

    Data is appended to datafile availability.csv at
    ~/Data/la_permanence/ in the format

    YYYY-MM-DD hh:mm:ss,x,y
    """
    page = get_LaPermanence_page()
    if page is None:
        return
    row = extract_data(page)
    write_data_to_file(row)


def combine_kaggle_local():
    """Download dataset from
    https://www.kaggle.com/antoinechoffrut/la-permanence, combine with
    locally collected dataset and save locally.

    The data collected by this machine is at
    ~/Data/la_permanence/availability.csv and the combined datafile is
    at ~/kaggle/la_permanence/availability.csv

    """
    logger = logging.getLogger()

    logger.info("Load local dataset...")
    dm_local = DataManager(source='local')

    logger.info("Load dataset from kaggle...")
    dm_kaggle = DataManager(source='kaggle')

    if dm_kaggle is None:
        logger.error("Cannot combine with local dataset.")
        return 1

    logger.info("Combine datasets...")
    dm_combined = dm_kaggle.merge(dm_local)

    filepath = os.path.join(COMBINED_DIR, FILENAME)

    logger.info(f"Save combined dataset to {filepath}")
    dm_combined.data.to_csv(filepath, index=True)

    return 0


def generate_new_figures():
    logger = logging.getLogger()
    # Parameters
    # Data parameters
    nrows = None
    resol = 1
    mode = 'WeekOf'

    # Plotting parameters
    alpha = 0.8
    color_mean = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]

    color_fill = color_mean
    alpha_fill = 0.4

    color_current_week = 'k'
    alpha_current_week = alpha

    color_previous_week = 'k'
    alpha_previous_week = alpha

    color_label = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]

    # Locations
    moulin = Location('moulin')
    alesia = Location('alesia')
    locations = [moulin, alesia]

    # Load data
    logger.info(f"Load data (nrows={nrows})...")
    dm = DataManager(nrows=nrows)
    dm.localize()

    dm.resample(resol=resol)

    last_timestamp = dm.data.index.max()
    last_time_string = last_timestamp.strftime('%Hh%M')
    last_date_string = last_timestamp.strftime('%d/%m/%Y')
    last_minute_of_week = \
        last_timestamp.dayofweek * ONE_DAY \
        + last_timestamp.hour * ONE_HOUR \
        + last_timestamp.minute

    past_week_dm = dm.past_week()

    for location in locations:
        logger.debug(f"Location: {location.name}")
        view = dm.create_view(location=location, mode=mode)

        view_mean = view.mean()

        view_std = view.std()

        view_lower = view_mean - view_std
        view_lower.clip(0, view.location.max_seats, inplace=True)

        view_upper = view_mean + view_std
        view_upper.clip(0, view.location.max_seats, inplace=True)

        views = view.split_columns()

        past_week = \
            past_week_dm.create_view(
                location=location,
                mode=mode
            ).split_columns()

        current_week = past_week[-1]

        if len(views) > 1:
            previous_week = past_week[-2]
        else:
            previous_week = None

        if location.name == 'moulin':
            figsize = (14, 10)
        elif location.name == 'alesia':
            figsize = (14, 12)

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize)

        title = f"Disponibilité à {last_time_string} ({last_date_string})"

        xlabel = ""  # view_week.row_label()
        xticks, xticklabels = view.row_ticks(plot_type='plot')
        ylabel = "Nombre de places"  # 'Available seats'
        yticks, yticklabels = view.location.seat_ticks()

        for ax in (ax_top, ax_bot):

            ax.fill_between(
                    view_mean.table.index,
                    view_upper.table.values.flatten(),
                    view_lower.table.values.flatten(),
                    alpha=alpha_fill,
                    color=color_fill,
                    linewidth=0
                )

            ax.grid()

            ax.set_xticks(xticks)
            ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.tick_params(axis='y', labelrotation=0)
            ax.set_ylim([yticks[0], yticks[-1]])
            ax.set_xlim([0, ONE_WEEK])

        # Specific to top subplot
        if previous_week:
            handle_previous_week, = \
                ax_top.plot(previous_week.table,
                            color=color_previous_week,
                            alpha=alpha_previous_week)

        handle_current_week, = \
            ax_bot.plot(current_week.table,
                        color=color_current_week,
                        alpha=alpha_current_week)

        ax_top.set_title(title, fontsize=TITLE_FONTSIZE)
        ax_top.set_xticklabels(['']*len(xticks))

        # Specific to bottom subplot
        ax_bot.set_xlabel(xlabel, fontsize=18)
        ax_bot.set_xticklabels(xticklabels)

        ax_bot.tick_params(axis='x', labelrotation=45)

        ax_bot.plot(
            [last_minute_of_week, last_minute_of_week],
            [yticks[0], yticks[-1]],
            color=color_label,
            linewidth=4
        )
        horizontalshift = 4 * ONE_HOUR
        if last_minute_of_week < ONE_WEEK // 2:
            horizontalalignment = 'left'
        else:
            horizontalalignment = 'right'
            horizontalshift = -horizontalshift
        ax_bot.text(
            last_minute_of_week + horizontalshift,
            location.max_seats - 10,  # last_availability,
            last_timestamp.strftime('%d/%m/%Y %Hh%M'),
            color=color_label,
            fontweight='bold',
            horizontalalignment=horizontalalignment,
            fontsize=YLABEL_FONTSIZE
        )

        figname = os.path.join(
            FIG_DIR,
            '-'.join([location.name, 'summary']) + '.png'
        )
        logger.info(f'Save figure to {figname}...')
        plt.savefig(figname)

    return 0


# Classes
class Location:
    """Convenience class to store basic information on coworking space
    locations (rue du Fer à Moulin and rue d'Alésia)."""

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
        """Return ticks and ticklabels for axis showing number of
        availability seats at coworking space.  The range depends on
        the location, and an extra tick is displayed to show maximum
        number of seats at location."""
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
    """Mostly a convenience class to load datasets and create views
    (see other class View).

    Attributes:

    - data: pandas.DataFrame indexed by timestamp with number of
    available seats at locations in columns `moulin` and `alesia`.
    Further columns with temporal features are added to generate
    views.

    """

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
        """Load dataset as pandas.DataFrame.

        If `data` is not None, return `data` itself.
        If `source` is 'local', load locally collected dataset from
        file in ~/Data/la_permanence/availability.csv.
        If `source` is 'kaggle', download dataset from
        https://www.kaggle.com/antoinechoffrut/la-permanence.
        If `source` is 'combined', load dataset from file in
        ~/kaggle/la_permanence/availability.csv.

        Return: dataset as pandas.DataFrame `data`.
        """

        logger = logging.getLogger()
        if data is not None:
            return data.copy()

        if source == 'local':
            path = os.path.join(LOCAL_DIR, FILENAME)
        elif source == 'combined':
            path = os.path.join(COMBINED_DIR, FILENAME)
        elif source == 'kaggle':
            returncode, outs, errs = download_from_kaggle()
            if returncode != 0:
                logger.error(
                    "Could not download dataset from kaggle."
                )
                return None
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

        is_valid = cls.is_datafile_valid(path)

        if not is_valid:
            logger.error(f"Datafile {path} invalid.")
            return None

        data = pd.read_csv(
            path,
            parse_dates=['timestamp'],
            date_parser=cls.dateparse,
            nrows=nrows)

        data.set_index('timestamp', inplace=True)

        return data

    @staticmethod
    def is_datafile_valid(path):
        with open(path, 'r') as file:
            first_line = file.readline()

        columns = first_line.split('\n')[0].split(',')
        is_valid = \
            ('timestamp' in columns) and \
            ('moulin' in columns) and \
            ('alesia' in columns)

        return is_valid

    @staticmethod
    def dateparse(timestamp):
        """Date parser for timestamp in availability.csv."""
        return pd.datetime.strptime(timestamp, TIMESTAMP_FORMAT)

    def localize(self):
        logger = logging.getLogger()
        data = self.data

        if set(data.columns) != set(('moulin', 'alesia')):
            logger.warning(
                "No localization: temporal features already generated."
            )
            return 1

        if data.index.dtype == DTYPE_NONE:
            data.reset_index(inplace=True)
            data['timestamp'] = \
                data['timestamp'].apply(
                    lambda dt: dt.tz_localize(TZ_UTC).tz_convert(TZ_PARIS)
                )
            data.set_index("timestamp", inplace=True)
            return 0
        elif data.index.dtype == DTYPE_PARIS:
            logger.warning(
                "Index already timezone-aware in Paris timezone"
            )
            return 0
        else:
            logger.warning(
                "Index of unknown type, no localization performed."
            )
            return 1

    def unlocalize(self):
        logger = logging.getLogger()
        data = self.data

        if set(data.columns) != set(('moulin', 'alesia')):
            logger.warning(
                "No localization: temporal features already generated."
            )
            return 1

        if data.index.dtype == DTYPE_PARIS:
            data.reset_index(inplace=True)
            data['timestamp'] = \
                data['timestamp'].apply(
                    lambda dt: dt.tz_convert(TZ_UTC).tz_localize(None)
                )
            data.set_index("timestamp", inplace=True)
            return 0
        elif data.index.dtype == DTYPE_NONE:
            logger.warning(
                "Index already naive UTC"
            )
            return 0
        else:
            logger.warning(
                "Index of unknown type, no unlocalization performed."
            )
            return 1

    def resample(self, resol=5):
        """
        Resample attribute `data` at regular intervals of `resol`
        minutes.
        """
        self.data = \
            self.data.resample(
                f'{resol}T'
            ).mean().interpolate().round().astype(np.uint8)

    def augment_features(self):
        """Return dataframe with additional temporal features based on
        `timestamp`.
        These features are used in particular to generate views.
        """
        df = self.data.copy()

        df.reset_index(inplace=True)

        df['Date'] = \
            df['timestamp'].apply(
                lambda dt: dt.replace(hour=0, minute=0, second=0)
            )

        attributes = \
            ['Minute', 'Hour', 'Day', 'DayOfWeek', 'WeekOfYear', 'DayOfYear']
        for attr in attributes:
            df[attr] = getattr(df['timestamp'].dt, attr.lower())

        df['WeekOf'] = df['Date'] - df['DayOfWeek']*pd.Timedelta(days=1)

        df['MinuteOfDay'] = 60*df['Hour'] + df['Minute']

        df['MinuteOfWeek'] = 24*60*df['DayOfWeek'] + df['MinuteOfDay']

        for col in ['Date', 'WeekOf']:
            df[col] = pd.to_datetime(df[col].dt.date)
        df.set_index('timestamp', inplace=True)

        return df

    def create_view(self, location, mode):
        """Generate an object of class View according to `location`
        and `mode`.

        The parameter `location` is a `Location` object, corresponding
        to either `rue du Fer à Moulin` or `rue d'Alésia.

        The parameter `mode` is one of:
        - 'DayOfWeek': Monday through Sunday;
        - 'DayOfYear': 1 through 365;
        - 'Date': date of day, e.g. day with date 2019-01-08;
        - 'WeekOfYear': 1 through 52;
        - 'WeekOf': week starting on a Monday, e.g. 2019-05-27
        mode = mode
        represents the week commencing on Monday 27 May 2019.

        """
        index = DataManager.MODE_TO_INDEX[mode]

        df = \
            self.augment_features().loc[:, [location.name, mode, index]]

        df.rename(columns={location.name: 'availability'}, inplace=True)
        table = pd.pivot_table(
            df,
            index=index,
            columns=mode,
            values='availability'
        )

        return View(table, location)

    def past_week(self):
        """Generate DataManager object with data from last 7*24 hours."""
        last_timestamp = self.data.index.max()
        last_week_timestamp = last_timestamp - pd.Timedelta('7D')
        last_week = self.data[self.data.index >= last_week_timestamp].copy()

        return DataManager(last_week)

    def merge(self, other):
        """Generate new DataManager object from another by merging
        their dataframes."""
        logger = logging.getLogger()
        df_left = self.data.copy()
        df_right = other.data.copy()

        if df_left.index.dtype != df_right.index.dtype:
            logger.error("Dataframes have index of different types.")
            return None

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
        df.sort_values(by='timestamp', inplace=True)
        df.drop_duplicates(subset='timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)

        return DataManager(data=df)


class View:
    """Convenience class to display data in a calendar view.

    Columns are indexed by mode:
    - 'DayOfWeek': Monday through Sunday;
    - 'DayOfYear': 1 through 365;
    - 'Date': date of day, e.g. day with date 2019-01-08;
    - 'WeekOfYear': 1 through 52;
    - 'WeekOf': week starting on a Monday, e.g. 2019-05-27
    represents the week commencing on Monday 27 May 2019.

    If mode is 'DayOfWeek', 'DayOfYear', or 'Date', the rows are
    indexed by 'MinuteOfDay' (minimum: 0, maximum: 1439).
    If mode is 'WeekOfYear' or 'WeekOf', the rows are indexed by
    'MinuteOfWeek' (minimu: 0, maximu: 10079).

    The index name is 'MinuteOfDay' or 'MinuteOfWeek' accordingly.
    The mode is recordedin the columns name (not to be confused with
    column labels of the dataframe).

    A View object has three (should be two) attributes:
    - `table`: a pandas.DataFrame giving calendar view of the data;
    - `location`: the Location object corresponding to the coworking
    space location;
    - `index`: (todo: redundant)
    """
    def __init__(self, table, location):
        self.table = table
        # self.index = self.table.index.name
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

        views = [View(table, self.location)
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
            (self.table.columns.name == other.table.columns.name) \
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
        if self.table.index.name == 'MinuteOfDay':
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][0:7]
        if self.table.index.name == 'MinuteOfWeek':
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][0:1]

    def legend_labels(self):
        if self.table.index.name == 'MinuteOfDay':
            return DAYS_OF_THE_WEEK.values()
        if self.table.index.name == 'MinuteOfWeek':
            return []

    def row_label(self):
        if self.table.index.name == 'MinuteOfDay':
            return 'Time of day'
        if self.table.index.name == 'MinuteOfWeek':
            return 'Day of week'

    def row_ticks(self, plot_type):
        if self.table.index.name == 'MinuteOfDay':
            ticks = HOURLY
            ticklabels = \
                ["{0:02d}h00".format(x)
                 for x in range(0, 24)]

        elif self.table.index.name == 'MinuteOfWeek':
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

    # logger = logging.getLogger()
    # download_from_kaggle()
    # git_push_figures()
    


if __name__ == '__main__':
    main()
