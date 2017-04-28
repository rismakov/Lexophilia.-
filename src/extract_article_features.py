from __future__ import division
import pandas as pd
from stylometry_analysis import StyleFeatures
import json
from datetime import datetime

NONARTICLE_WORDS = ['270', '480', 'tsgp', 'experienceID', 'api', 'setsize',
                    'containing', 'ID:', 'Getty', 'Images', 'iframe',
                    'video-cdn.buzzfeed.com', 'Video', 'Press',
                    'available', 'AFP', 'UPDATE', 'BuzzFeed', '})',
                    'Photo', 'Slate.', 'Image', 'Advertisement', 'taps',
                    'cover', 'size', ' am', 'pm', 'homepage', 'js', 'webkit',
                    'This article is from the archive of our partner The Wire',
                    'This article is from the archive of our partner',
                    'READ ARTICLE', '@theatlantic.com',
                    'SIGN UP FOR OUR NEWSLETTER', 'windowwidth']

with open('text_files/countries.txt') as f:
    COUNTRIES = set(f.readlines())


def remove_words(string, words):
    for word in words:
        string = string.replace(word, '')
    return string


def convert_dates_to_dt(dates, date_format):
    ok_formats = ['%B %d %Y', '%b %d, %Y', '%d %b %Y', '%Y-%m-%d']

    dates_formatted = []
    for date in dates:
        if date.split()[0] == 'Sept.':
            date = date.replace('t', '')
        elif date.split()[0] == 'Updated:':
            date = ' '.join(date.split()[1:4])

        if date_format in ok_formats or '.' in date.split()[0]:
            frmt = date_format
        elif ',' in date.split()[1]:
            frmt = '%B %d, %Y'
        else:
            frmt = '%B %d %Y'

        dates_formatted.append(datetime.strptime(date, frmt))
    return dates_formatted


def get_features(articles):
    all_features = []
    for i, article in enumerate(articles):
        features = StyleFeatures(article)
        all_features.append(features)

    return all_features


def add_title_countries_to_df(df):
    df['countries_in_title'] = [[country for country in COUNTRIES
                                if country in title] for title in df.title]
    return df


def add_all_features_to_df(df):
    df['style_features'] = get_features(df.article)

    df['article_len'] = [dp['article_len'] for dp in df.style_features]
    df = df[df.article_len > 70]  # remove datapoints where article_len is less than 70 words
    for feature in df['style_features'][3]:
        df[feature] = [dp[feature] for dp in df.style_features]
    df = add_title_countries_to_df(df)

    return df


def create_and_save_feature_df(df, date_format, newssite):
    print '___________________{}_____________________ '.format(newssite)
    print '{} articles scraped'.format(len(df))

    df['article'] = df.article.apply(lambda x: remove_words(x, NONARTICLE_WORDS))
    df['date_posted'] = convert_dates_to_dt(df.date_posted, date_format)

    print 'adding style features to {} dataframe....'.format(newssite)
    df = add_all_features_to_df(df)
    print 'finished adding features to {} data'.format(newssite)

    df.to_json(path_or_buf='feature_data/{}_features.json'.format(newssite))

if __name__ == "__main__":
    nyt_data = pd.read_json('data/nyt_data.json')
    nyt_data['date_posted'] = [date[:10] for date in nyt_data.date_posted]
    nyt_data = nyt_data[nyt_data.article != '']

    # nyt = nyt_data[nyt_data.section != 'Opinion']
    nyt_op = nyt_data[nyt_data.section == 'Opinion']
    nyt_op = nyt_op.reset_index()
    nyt_op.drop('index', axis=1, inplace=True)
    # create_and_save_feature_df(nyt, '%Y-%m-%d','nyt')
    create_and_save_feature_df(nyt_op, '%Y-%m-%d', 'nyt_op')

    '''
    time_op_data = pd.read_json('data/time_opinion_data.json')
    time_op_data = time_op_data[time_op_data.date_posted != '']
    create_and_save_feature_df(time_op_data, '%b %d, %Y','time_opinion')

    bb = pd.read_json('data/breitbart_data.json')
    create_and_save_feature_df(bb, '%d %b %Y','breitbart')

    #slate
    slate = pd.read_json('data/slate_data.json')
    slate['date_posted'] = [' '.join(date.split()[:3]) for date in slate.date_posted]
    create_and_save_feature_df(slate, '%b. %d %Y','slate')

    #buzzfeed
    bf = pd.read_json('data/buzzfeed_data.json')
    bf['date_posted'] = [date_code.split('(')[4] for date_code in bf.date_posted]
    bf['date_posted'] = [''.join([char for char in d if char.isdigit()]) for d in bf.date_posted]
    bf['date_posted'] = [datetime.fromtimestamp(int(dt)).strftime('%B %d %Y') for dt in bf.date_posted]
    create_and_save_feature_df(bf, '%B %d %Y', 'buzzfeed')
    '''
    '''
    #time
    time_data = pd.read_json('data/time_opinion_data.json')
    time_data = time_data[time_data.date_posted != ''].iloc[3:,:]
    create_and_save_feature_df(time_data, '%b. %d, %Y','time')
    '''
    '''
    #atlantic
    atlantic_data = pd.read_json('data/atlantic_data.json')
    atlantic_data['date_posted'] = [date.replace('\n','') for date in atlantic_data['date_posted']]
    create_and_save_feature_df(atlantic_data, '%b %d, %Y','atlantic')
    '''
