from __future__ import division

import json
import pickle
import requests

from bs4 import BeautifulSoup
from string import punctuation


TYPE = ['Article', 'Blog', 'News', 'An Analysis; News Analysis', 'Text']
SECTIONS = ['World', 'Opinion', 'Blogs', 'U.S.']


def get_soup(url):
    r = requests.get(url)
    return BeautifulSoup(r.text, 'html.parser')


def get_article_from_url(url, klass):
    soup = get_soup(url)
    code_text = soup.article.find_all(class_=klass)

    article = ''
    for code in code_text:
        article = article + code.text
    return article


def get_nyt_datapoint_info(nyt_data, filename, datapoints=[]):

    for i, dp in enumerate(nyt_data):
        print 'data number: {}'.format(i)
        dp = dp['response']

        for i2, doc in enumerate(dp['docs']):

            if doc['type_of_material'] in TYPE and doc['section_name'] in SECTIONS: 
                datapoint = {}

                datapoint['section'] = doc['section_name']
                datapoint['title'] = doc['headline']
                datapoint['date_posted'] = doc['pub_date']
                datapoint['article_len'] = doc['word_count']
                datapoint['author'] = doc['byline']

                try:
                    datapoint['article'] = get_article_from_url(doc['web_url'], 'story-body-text story-content')

                    print 'doc number: {}'.format(i2)
                    datapoints.append(datapoint)
                    save_to_json('nyt_data.json', datapoints)
                except:
                    pass

    return datapoints


def open_json_file(filename):
    json_data = open(filename).read()
    return json.loads(json_data)


def save_to_json(filename, datapoints):
    with open(filename, 'w') as outfile:
        json.dump(datapoints, outfile)

if __name__ == '__main__':
    nyt_data = open_json_file('data/nyt_archive_start_2010.json')
    datapoints = get_nyt_datapoint_info(nyt_data, 'data/nyt_data.json')
    save_to_json('data/nyt_data.json', datapoints)
