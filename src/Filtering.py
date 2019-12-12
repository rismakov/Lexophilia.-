import numpy as np
import pandas as pd

from datetime import datetime

from Countries import count_country_mentions


def filter_data(df, filter_date=False, min_year=None):
    df['date_posted'] = df.date_posted.apply(lambda date:
                                             datetime.fromtimestamp(date/1000))
    df['year'] = [date.year for date in df.date_posted]
    if filter_date:
        df = df[df.year >= min_year]
    # df = add_title_countries_to_df(df)
    return df


def filter_out_unknowns(data):
    return data[data.author_gender != 'unknown']


def add_midpoint_date_to_df(df, grouped_df):
    '''
    Adds the midpoint date of the grouped data to grouped dataframe.

    Input: dataframe, grouped dataframe
    Output: grouped dataframe with additional datetime column
    '''
    average_dates = []
    for author in grouped_df.index:
        all_dates_of_author = df[df.author==author]['date_posted']
        average_dates.append(all_dates_of_author.min() +
                             (all_dates_of_author.max() -
                             all_dates_of_author.min())/2)

    grouped_df['average_date'] = average_dates
    return grouped_df


def group_by_month(df):
    grouped_by_time = df.set_index('average_date')
    return grouped_by_time.resample('5m').mean()


def group_by_author(df, print_gender_info=False):
    '''
    Groups dataframe by author and takes mean of numeric values and mode of
    non-numeric values of each group. Adds midpoint data (last article minus
    first article date) of each author to dataframe.

    Input: dataframe, True False whether you want to additionally print gender info
    Output: Grouped dataframe with author name as index.
    '''
    mean_df = df.groupby(['author']).mean()
    mode_df = df.groupby(['author']).agg(lambda x: x.value_counts().index[0])
    mode_df = mode_df[mode_df.columns.difference(mean_df.columns)]
    grouped_df = pd.concat([mean_df, mode_df], axis=1)
    grouped_df = add_midpoint_date_to_df(df, grouped_df)

    grouped_df['year'] = [int(num) for num in grouped_df.year]

    if print_gender_info:
        print_gender_percentage(grouped_df)

    return grouped_df


def seperate_by_gender(df):
    return df[df.author_gender == 'female'], df[df.author_gender == 'male']
