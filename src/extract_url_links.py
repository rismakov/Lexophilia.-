from __future__ import division

import pickle
import requests
import unicodedata

from bs4 import BeautifulSoup


def convert_unicode_to_str(unicode):
    return unicodedata.normalize('NFKD', unicode).encode('ascii', 'ignore')


def get_reuters_urls_from_url_bases(urls):
    all_urls = []
    for url in urls:
        if url[:44] == 'http://www.reuters.com/investigates/section/':
            inner_urls = get_list_of_all_urls(url, 'section-article row col-md-11 col-md-offset-1 col-lg-9 col-lg-offset-1', '', xrange(1))
            for inner_url in inner_urls:
                all_urls.append(inner_url)
        else:
            all_urls.append(url)
    return all_urls


def get_list_of_vox_urls(url_base, page_nums, urls=[]):

    for i in page_nums:  # goes through pages
        url = url_base.format(i+1)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        code_sections = soup.find_all('h3')

        for code in code_sections:  # goes through each link
            url = convert_unicode_to_str(code.a['href'])
            urls.append(url)

        save_urls_to_pickle('time_urls', urls)

    return urls


def get_list_of_all_urls(url_base, klass, klass2, page_nums, urls=[]):

    for i in page_nums:  # goes through pages
        print i
        url = url_base.format(i+1)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        code_sections = soup.find_all(class_=klass)
        if klass2:
            code_sections = code_sections[1].find_all(class_=klass2)  # was [1] before, [0] for slate

        for code in code_sections:  # goes through each link
            try:
                url = convert_unicode_to_str(code.a['href'])
                urls.append(url)
            except:
                pass

        save_urls_to_pickle('time_opinion_urls',urls)

    return urls


def save_urls_to_pickle(file_name, urls):
    with open(file_name, 'wb') as fp:
        pickle.dump(urls, fp)


def open_urls(file_name):
    with open(file_name, 'rb') as outfile:
        return pickle.load(outfile)

if __name__ == "__main__":
    '''
    breitbart_url_base = 'http://www.breitbart.com/big-government/page/{}/'
    klass = 'title'
    breitbart_urls = get_list_of_all_urls(breitbart_url_base,klass,'',xrange(2489))
    save_urls_to_pickle('urls/breitbart_urls', breitbarts_urls)
    print 'retrieved all {0} Breitbart Post url links'.format(len(breitbart_urls))
    '''
    '''
    huff_url_base = 'http://www.huffingtonpost.com/section/politics?page={}'
    klass = 'card__link'
    huff_urls = get_list_of_all_urls(huff_url_base,klass,'',xrange(15))
    save_urls_to_pickle('huff_urls', huff_urls)
    print 'retrieved all {0} Huffington Post url links'.format(len(huff_urls))


    vox_url_base = 'http://www.vox.com/world/archives/{}'
    vox_urls = get_list_of_vox_urls(vox_url_base,xrange(1))
    save_urls_to_pickle('vox_urls', vox_urls)
    print 'retrieved all {0} Buzzfeed url links'.format(len(bf_urls))
    '''
    '''
    # buzzfeed
    buzzfeed_url_base = 'https://www.buzzfeed.com/politics?p={0}&z=5AIR6U&r=1'
    bf_klass = 'lede__title lede__title--medium'
    bf_urls = get_list_of_all_urls(buzzfeed_url_base,bf_klass,'',xrange(599))
    save_urls_to_pickle('buzzfeed_urls', bf_urls)
    print 'retrieved all {0} Buzzfeed url links'.format(len(bf_urls))
    '''

    # time world
    # time_url_base = 'http://time.com/world/page/{0}/'
    time_url_base = 'http://time.com/opinion/page/{}/'
    time_klass = 'section-archive-list'
    time_klass2 = 'section-article-title'
    # time_urls = open_urls('time_urls')
    time_urls = get_list_of_all_urls(time_url_base, time_klass, time_klass2,
                                     xrange(139), [])
    save_urls_to_pickle('time_opinion_urls', time_urls)
    print 'retrieved all {0} Time url links'.format(len(time_urls))

    '''
    # reuters investigates- special reports
    reuters_url_base = 'http://www.reuters.com/investigates/section/reuters-investigates-201{}/'
    reuters_klass = 'item'
    #reuters_urls = get_list_of_all_urls(reuters_url_base,reuters_klass,'',[6,5,4,3])
    reuters_urls = open_urls('urls/reuters_urls')
    reuters_all_urls = get_reuters_urls_from_url_bases(reuters_urls)
    save_urls_to_pickle('urls/reuters_full_urls', reuters_all_urls)
    print 'retrieved all {0} Reuters url links'.format(len(reuters_all_urls))

    #the atlantic news
    atlantic_base_url = 'https://www.theatlantic.com/latest/?page={}'
    atlantic_klass = 'article blog-article '
    atlantic_urls = get_list_of_all_urls(atlantic_base_url, atlantic_klass, '', xrange(8673))
    save_urls_to_pickle('atlantic_urls', atlantic_urls)
    print 'retrieved all {0} Atlantic url links'.format(len(atlantic_urls))

    # slate politics and international news: 149, Feb 2001
    slate_url_base = 'http://www.slate.com/articles/news_and_politics/politics.{0}.html'
    slate_klass = 'tiles'
    slate_klass2 = 'tile long-hed stacked'
    slate_urls = get_list_of_all_urls(slate_url_base,slate_klass,slate_klass2,xrange(60))
    slate_urls2 = get_list_of_all_urls(slate_url_base,'tiles','tile basic',xrange(60,150))

    all_slate_urls = slate_urls + slate_urls2

    extract_and_save_data('slate_data.json', get_slate_datapoint_info,slate_urls)
    '''
