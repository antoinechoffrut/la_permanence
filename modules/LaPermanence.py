#!/usr/bin/env python
"""
Main module containing classes and functions for La Permanence
project.
"""

# Imports
import os
import pytz
# import datetime
import numpy as np
import pandas as pd

# Plotting libraries for testing purposes only
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


# Constants

ONE_HOUR = 60
HOURLY = [h * ONE_HOUR for h in range(0, 24)]

ONE_DAY = 24*ONE_HOUR
DAILY = [d * ONE_DAY for d in range(0, 7)]

ONE_WEEK = 7*ONE_DAY
WEEKLY = [w * ONE_WEEK for w in range(0, 52)]


# Functions


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
    LOCATION_NAMES = LOC_INFO.keys()

    def __init__(self, name):
        if name not in self.LOC_INFO.keys():
            print(f'ERROR: invalid location name "{name}"')
        for key in self.LOC_INFO[name].keys():
            setattr(self, key, self.LOC_INFO[name][key])

    def ticks(self):
        ticks = \
            list(range(0, 10 * (2 + self.max_seats // 10), 10)) \
            + [self.max_seats]
        ticklabels = [str(n) for n in ticks[:-1]] + [f'max: {ticks[-1]}']
        z = zip(ticks, ticklabels)
        z = sorted(z, key=lambda x: x[0])
        ticks, ticklabels = zip(*z)
        return ticks, ticklabels


class DataManager:
    FILENAME = 'availability.csv'
    PATH = os.path.join(
        os.path.expanduser('~'), 'kaggle', 'la_permanence'
    )

    TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

    TZ_UTC = pytz.timezone("UTC")
    TZ_PARIS = pytz.timezone("Europe/Paris")

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

    # DAYS_OF_THE_WEEK = {0: 'Monday',
    #                     1: 'Tuesday',
    #                     2: 'Wednesday',
    #                     3: 'Thursday',
    #                     4: 'Friday',
    #                     5: 'Saturday',
    #                     6: 'Sunday'}
    DAYS_OF_THE_WEEK = {0: 'lundi',
                        1: 'mardi',
                        2: 'mercredi',
                        3: 'jeudi',
                        4: 'vendredi',
                        5: 'samedi',
                        6: 'dimanche'}

    @staticmethod
    def dateparse(timestamp, timestamp_format=TIMESTAMP_FORMAT):
        return pd.datetime.strptime(timestamp, timestamp_format)

    @classmethod
    def load_data(cls, data=None, filename=FILENAME, path=PATH, nrows=None):
        if data is not None:
            return data.copy()

        path_to_file = os.path.join(path, filename)

        if not os.path.exists(path_to_file):
            print(f"Error: no file at {path_to_file}.")
            return None

        with open(path_to_file, 'r') as file:
            first_line = file.readline()

        columns = first_line.split('\n')[0].split(',')
        valid_columns = \
            ('timestamp' in columns) and \
            (('moulin' in columns) or ('alesia' in columns))
        if not valid_columns:
            print("WARNING: incorrect column names: {columns}.")
            return None
        data = pd.read_csv(
            path_to_file,
            parse_dates=['timestamp'],
            date_parser=cls.dateparse,
            nrows=nrows)

        data['timestamp'] = \
            data['timestamp'].apply(
                lambda dt: dt.tz_localize(cls.TZ_UTC).tz_convert(cls.TZ_PARIS)
            )

        data.set_index("timestamp", inplace=True)

        return data

    def resample(self, resol=5):
        self.data = \
            self.data.resample(
                f'{resol}T'
            ).mean().interpolate().round().astype(np.uint8)

    def __init__(self, data=None, filename=FILENAME, path=PATH, nrows=None):
        self.data = self.load_data(data, filename, path, nrows)

    def augment_features(self):
        data = self.data

        data.reset_index(inplace=True)

        data['Date'] = data['timestamp'].apply(
            lambda ts: pd.Timestamp(
                ts.year, ts.month, ts.day
            ).tz_localize(self.TZ_PARIS)
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
        df.rename(columns={location.name: 'availability'}, inplace=True)
        table = pd.pivot_table(
            df,
            index=index,
            columns=mode,
            values='availability'
        )
        freq = {
            'n': self.data.index.freq.n,
            'name': self.data.index.freq.name,
        }

        return View(table, location, freq)

    def split_by_mode(self, mode, location):
        if mode not in self.MODES.keys():
            print("WARNING: valid parameters are day and week.")
            return None
        data = self.data

        index = self.MODE_TO_INDEX[mode]

        dataframes = [
            data.loc[data[mode] == mode_value, [location.name, index]]
            for mode_value in data[mode].unique()
        ]
        tables = [
            pd.DataFrame(
                data=dataframe[location.name].values,
                index=dataframe[index],
                columns=['availability']
            )
            for dataframe in dataframes
        ]

        freq = {
            'name': data.index.freq.name,
            'n': data.index.freq.n
        }

        views = [
            View(table, location, freq)
            for table in tables
        ]

        return views

    def past_week(self):
        last_timestamp = self.data.index.max()
        last_week_timestamp = last_timestamp - pd.Timedelta('7D')
        last_week = self.data[self.data.index >= last_week_timestamp].copy()

        return DataManager(last_week)


class View:

    def __init__(self, table, location, freq):
        self.table = table
        self.index = self.table.index.name
        self.mode = self.table.columns.name
        self.freq = freq
        self.location = location

    def apply_func(self, func):
        new_table = \
            self.table.apply(func, axis=1).to_frame(name=func.__name__)
        new_view = View(new_table, self.location, freq=self.freq)
        return new_view

    def mean(self):
        table = self.table.mean(axis=1).to_frame(name='mean')
        location = self.location
        freq = self.freq
        return View(table, location, freq)

    def std(self):
        table = self.table.std(axis=1).to_frame(name='std')
        location = self.location
        freq = self.freq
        return View(table, location, freq)

    def min(self):
        table = self.table.min(axis=1).to_frame(name='min')
        location = self.location
        freq = self.freq
        return View(table, location, freq)

    def max(self):
        table = self.table.max(axis=1).to_frame(name='max')
        location = self.location
        freq = self.freq
        return View(table, location, freq)

    def clip(self, lower=None, upper=None, *args, **kwargs):
        location = self.location
        freq = self.freq
        if ('inplace' in kwargs.keys()) and (kwargs['inplace'] is True):
            self.table.clip(lower=lower, upper=upper, axis=1, *args, **kwargs)
            return None
        else:
            table = self.table.clip(
                lower=lower, upper=upper,
                axis=1, *args, **kwargs)
            return View(table, location, freq)

    def compatible_with(self, other):
        res = \
            (self.index == other.index) \
            and (self.mode == other.mode) \
            and (self.freq == other.freq) \
            and (self.location == other.location)
        return res

    def __add__(self, other):
        if not self.compatible_with(other):
            print("WARNING: views not compatible")
            return self
        values = self.table.values + other.table.values
        table = pd.DataFrame(data=values, index=self.table.index)
        location = self.location
        freq = self.freq
        return View(table, location, freq)

    def __sub__(self, other):
        if not self.compatible_with(other):
            print("WARNING: views not compatible")
            return self
        values = self.table.values - other.table.values
        table = pd.DataFrame(data=values, index=self.table.index)
        location = self.location
        freq = self.freq
        return View(table, location, freq)

    def __rmul__(self, scalar):
        return View(scalar*self.table, self.location, self.freq)

    def color_cycle(self):
        if self.index == 'MinuteOfDay':
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][0:7]
        if self.index == 'MinuteOfWeek':
            return plt.rcParams['axes.prop_cycle'].by_key()['color'][0:1]

    def legend_labels(self):
        if self.index == 'MinuteOfDay':
            return DataManager.DAYS_OF_THE_WEEK.values()
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
                [DataManager.DAYS_OF_THE_WEEK[int(x // ONE_DAY)].ljust(12)
                 if (x % ONE_DAY == 0) and (x < ONE_WEEK)
                 else f'{(x % ONE_DAY) // ONE_HOUR:02d}h'
                 for x in ticks]

        if (plot_type == 'heatmap') \
           and self.freq['name'] == 'T':
            ticks = [x // self.freq['n'] for x in ticks]

        return ticks, ticklabels

    def col_label(self):
        return DataManager.MODES[self.mode]

    def col_ticks(self, plot_type):

        if self.mode == 'Date':
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

        if self.mode == 'DayOfYear':
            vals = self.table.columns.values
            enum = enumerate(vals)
            ticks, ticklabels = \
                zip(*[
                    (idx + 0.5, val)
                    for idx, val in enum
                    if idx % 7 == 0
                ])

            return ticks, ticklabels

        if self.mode == 'DayOfWeek':
            vals = [the_day[0:3]
                    for the_day in DataManager.DAYS_OF_THE_WEEK.values()]
        elif self.mode == 'WeekOf':
            date_min = self.table.columns.min()
            date_max = self.table.columns.max()
            date_range = \
                pd.date_range(date_min, date_max, freq='W')
            vals = [dt.strftime('Week of %Y-%m-%d')
                    for dt in self.table.columns]
        elif self.mode == 'WeekOfYear':
            vals = self.table.columns.values

        enum = enumerate(vals)
        ticks, ticklabels = \
            zip(*[
                (idx + 0.5, val)
                for idx, val in enum
            ])

        return ticks, ticklabels


def heatmap(view, figsize, savefig=False):
    fig, ax = plt.subplots(figsize=figsize)

    df = view.table.transpose()
    sns.heatmap(
        df,
        vmin=0,
        vmax=view.location.max_seats,
    )

    title = view.location.name
    xlabel = view.ro_wlabel()
    xticks, xticklabels = view.row_ticks(plot_type='heatmap')
    ylabel = view.col_label()
    yticks, yticklabels = view.col_ticks(plot_type='heatmap')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.tick_params(axis='x', labelrotation=45)

    ax.tick_params(axis='y', labelrotation=0)

    if savefig:
        plt.savefig(
            '-'.join([view.location.name, view.mode, 'heatmap']) + '.png'
        )


def plot(view, figsize, savefig=False):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(
        'color',
        view.color_cycle()
    )

    df = view.table
    plt.plot(
        df,
    )

    title = view.location.name
    xlabel = view.row_label()
    xticks, xticklabels = view.row_ticks(plot_type='plot')
    ylabel = 'Available seats'
    yticks, yticklabels = view.location.ticks()
    legend_labels = view.legend_labels()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.tick_params(axis='x', labelrotation=45)

    ax.tick_params(axis='y', labelrotation=0)

    plt.legend(labels=legend_labels)
    ax.grid()

    if savefig:
        plt.savefig(
            '-'.join([view.location.name, view.mode, 'plot']) + '.png'
        )


# Scenarios

def heatmap_grid_search(nrows=None, resol=5, figsize=(14, 8), savefig=True):

    # Locations
    moulin = Location('moulin')
    alesia = Location('alesia')
    locations = [moulin, alesia]

    # Location-by-mode
    location_mode = \
        [(location, mode)
         for location in locations
         for mode in DataManager.MODES.keys()]

    dm = DataManager(nrows=nrows)
    dm.resample()
    dm.augment_features()

    for location, mode in location_mode:
        print(f'Location: {location.name}.  Mode: {mode}.')
        view = dm.create_view(location, mode)

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            view.table.transpose(),
            vmin=0,
            vmax=view.location.max_seats,
        )

        title = view.location.address
        xlabel = view.row_label()
        xticks, xticklabels = view.row_ticks(plot_type='heatmap')
        ylabel = view.col_label()
        yticks, yticklabels = view.col_ticks(plot_type='heatmap')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(ylabel)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.tick_params(axis='x', labelrotation=45)

        ax.tick_params(axis='y', labelrotation=0)

        if savefig:
            plt.savefig(
                '-'.join([view.location.name, view.mode, 'heatmap']) + '.png'
            )


def summary(mode, nrows=None, resol=5, figsize=(14, 8), savefig=True):

    # Locations
    moulin = Location('moulin')
    alesia = Location('alesia')
    locations = [moulin, alesia]

    dm = DataManager(nrows=nrows)
    dm.resample(resol=resol)
    dm.augment_features()

    # Parameters

    z_range = [.2, .5, 1]

    color_mean = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    alpha_mean = 0.8

    color_fill = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]

    for location in locations:
        print(f'Location: {location.name}')

        view_week = dm.create_view(location, mode=mode)
        view_mean = view_week.mean()
        view_std = view_week.std()

        views_upper = \
            [(view_mean + z*view_std).clip(0, view_week.location.max_seats)
             for z in z_range]
        views_lower = \
            [(view_mean - z*view_std).clip(0, view_week.location.max_seats)
             for z in z_range]

        fig, ax = plt.subplots(figsize=figsize)

        for idx, z in enumerate(z_range):
            ax.fill_between(
                view_mean.table.index,
                views_upper[idx].table.values.flatten(),
                views_lower[idx].table.values.flatten(),
                alpha=1/len(z_range),
                color=color_fill,
                linewidth=0
            )
            handle_mean, = \
                ax.plot(view_mean.table, color=color_mean, alpha=alpha_mean)

        title = '-'.join([
            view_week.location.address.capitalize(),
        ])
        xlabel = view_week.row_label()
        xticks, xticklabels = view_week.row_ticks(plot_type='plot')
        ylabel = 'Available seats'
        yticks, yticklabels = view_week.location.ticks()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(ylabel)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim([yticks[0], yticks[-1]])

        ax.tick_params(axis='x', labelrotation=45)

        ax.tick_params(axis='y', labelrotation=0)

        plt.grid()

        savefig = True
        if savefig:
            plt.savefig(
                '-'.join([location.name, mode, 'distribution']) + '.png'
            )

    return


def basic_statistics(mode, nrows=None, resol=5, figsize=(14, 8), savefig=True):

    resol = 5

    # Locations
    moulin = Location('moulin')
    alesia = Location('alesia')
    locations = [moulin, alesia]

    dm = DataManager(nrows=nrows)
    dm.resample(resol=resol)
    dm.augment_features()

    # Parameters

    alpha = 0.8

    color_mean = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    alpha_mean = alpha

    color_min = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    alpha_min = alpha

    color_max = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
    alpha_max = alpha_min

    color_std = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    alpha_std = alpha_mean

    # Plotting
    for location in locations:
        print(f'Location: {location.name}')

        view_week = dm.create_view(location, mode=mode)
        view_mean = view_week.mean()
        view_std = view_week.std()

        view_max = view_week.max()
        view_min = view_week.min()

        fig, ax = plt.subplots(figsize=figsize)

        handle_max, = \
            ax.plot(view_max.table, color=color_max, alpha=alpha_max)
        handle_mean, = \
            ax.plot(view_mean.table, color=color_mean, alpha=alpha_mean)
        handle_min, = \
            ax.plot(view_min.table, color=color_min, alpha=alpha_min)

        handle_std, = \
            ax.plot(view_std.table, color=color_std, alpha=alpha_std)

        title = '-'.join([
            view_week.location.address.capitalize(),
        ])
        xlabel = view_week.row_label()
        xticks, xticklabels = view_week.row_ticks(plot_type='plot')
        ylabel = 'Available seats'
        yticks, yticklabels = view_week.location.ticks()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(ylabel)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim([yticks[0], yticks[-1]])

        ax.tick_params(axis='x', labelrotation=45)

        ax.tick_params(axis='y', labelrotation=0)

        ax.legend(handles=[handle_max,
                           handle_mean,
                           handle_min,
                           handle_std],
                  labels=['max', 'mean', 'min', 'std']
                  )
        plt.grid()

        savefig = True
        if savefig:
            plt.savefig(
                '-'.join([location.name, mode, 'week-statistics']) + '.png'
            )


def inspect_past_week(nrows=None, resol=5, figsize=(14, 8), savefig=True):
    mode = 'WeekOf'

    # Locations
    moulin = Location('moulin')
    alesia = Location('alesia')
    locations = [moulin, alesia]

    dm = DataManager(nrows=nrows)
    dm.resample(resol=resol)
    dm.augment_features()

    past_week_dm = dm.past_week()
    last_timestamp = past_week_dm.data.index.max()

    # Parameters
    alpha = 0.8
    color_mean = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    # alpha_mean = alpha

    color_fill = color_mean
    alpha_fill = 0.4

    color_current_week = 'k'
    alpha_current_week = alpha

    color_previous_week = 'k'
    alpha_previous_week = alpha

    color_label = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]

    for location in locations:
        print(f"Location: {location.name}.")
        view_week = dm.create_view(location=location, mode=mode)
        view_mean = view_week.mean()
        view_std = view_week.std()
        view_lower = view_mean - view_std
        view_lower.clip(0, view_week.location.max_seats, inplace=True)
        view_upper = view_mean + view_std
        view_upper.clip(0, view_week.location.max_seats, inplace=True)

        past_week = past_week_dm.split_by_mode(location=location, mode=mode)
        current_week = past_week[-1]

        if len(past_week) > 1:
            previous_week = past_week[0]
        else:
            previous_week = None

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize)

        title = '-'.join([
            view_week.location.address.capitalize(),

        ])
        xlabel = view_week.row_label()
        xticks, xticklabels = view_week.row_ticks(plot_type='plot')
        ylabel = 'Available seats'
        yticks, yticklabels = view_week.location.ticks()

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
            ax.set_ylabel(ylabel)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.tick_params(axis='y', labelrotation=0)
            ax.set_ylim([yticks[0], yticks[-1]])

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

        ax_top.set_title(title)
        ax_top.set_xticklabels(['']*len(xticks))

        # Specific to bottom subplot
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_xticklabels(xticklabels)

        ax_bot.tick_params(axis='x', labelrotation=45)

        last_minute_of_week = \
            current_week.table.index[-1]
        # last_availability = \
        #     current_week.table.loc[last_minute_of_week, 'availability']

        ax_bot.plot([last_minute_of_week, last_minute_of_week],
                    [yticks[0], yticks[-1]],
                    color=color_label)
        horizontalshift = 4 * 60 // view_mean.freq['n']
        if last_minute_of_week < ONE_WEEK // 2:
            horizontalalignment = 'left'
        else:
            horizontalalignment = 'right'
            horizontalshift = -horizontalshift
        ax_bot.text(
            last_minute_of_week + horizontalshift,
            location.max_seats + 1,  # last_availability,
            last_timestamp.strftime('%Y-%m-%d %H:%M'),
            color=color_label,
            fontweight='bold',
            horizontalalignment=horizontalalignment)

        savefig = True
        if savefig:
            plt.savefig(
                '-'.join([location.name, mode, 'past', 'week']) + '.png'
            )


# Main function

def main():
    nrows = 10_000
    resol = 5
    savefig = True
    # for func in [heatmap_grid_search, summary,
    #              basic_statistics, inspect_past_week]:
    #     print(func)
    #     func(nrows=nrows, resol=resol)

    # print(basic_statistics)
    # basic_statistics(mode='WeekOf', nrows=nrows, resol=resol, savefig=savefig)
    # print(basic_statistics)
    # basic_statistics(mode='Date', nrows=nrows, resol=resol, savefig=savefig)
    # print(summary)
    # summary(mode='WeekOf', nrows=nrows, resol=resol, savefig=savefig)
    # print(summary)
    # summary(mode='Date', nrows=nrows, resol=resol, savefig=savefig)
    print(heatmap_grid_search)
    heatmap_grid_search(nrows=nrows, resol=resol, savefig=savefig)
    # print(inspect_past_week)
    # inspect_past_week(nrows=nrows, resol=resol, savefig=savefig)


if __name__ == '__main__':
    main()
