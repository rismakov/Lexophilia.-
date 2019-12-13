from __future__ import division

import logging
import pickle
import requests
import unicodedata

from bs4 import BeautifulSoup

from news_site_html_info import NEWSITE_HTML_INFO

logging.getLogger().setLevel(logging.INFO)


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


def get_vox_urls(url_base, page_nums, urls=[]):
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
    for i in page_nums:  # iterates through pages
        logging.info('Page number: {}'.format(i))
        url = url_base.format(i)
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
                logging.info('Error occurred at url {}'.format(url))
                pass

    return urls


def save_urls_to_pickle(file_name, urls):
    with open(file_name, 'wb') as fp:
        pickle.dump(urls, fp)


def open_urls(file_name):
    with open(file_name, 'rb') as outfile:
        return pickle.load(outfile)


if __name__ == "__main__":
    for newssite, info in NEWSITE_HTML_INFO.items():
        logging.info('Extracting url links from {}'.format(newssite))

        urls = get_list_of_all_urls(
            url_base=info['base_url'], 
            klass=info['klass'],
            klass2=info.get('klass2', ''),
            page_nums=range(1, info['page_nums']),
        )

        # save_urls_to_pickle('../{}_urls'.format(), urls)

        logging.info(
            'Retrieved {} {} post url links'.format(len(urls), newsite)
        )
    
    '''
    vox_url_base = 'http://www.vox.com/world/archives/{}'
    vox_urls = get_vox_urls(vox_url_base,range(1))
    save_urls_to_pickle('vox_urls', vox_urls)
    print 'retrieved all {0} Buzzfeed url links'.format(len(bf_urls))

    # reuters investigates- special reports
    reuters_url_base = 'http://www.reuters.com/investigates/section/reuters-investigates-201{}/'
    reuters_klass = 'item'
    #reuters_urls = get_list_of_all_urls(reuters_url_base,reuters_klass,'',[6,5,4,3])
    reuters_urls = open_urls('urls/reuters_urls')
    reuters_all_urls = get_reuters_urls_from_url_bases(reuters_urls)
    save_urls_to_pickle('urls/reuters_full_urls', reuters_all_urls)
    print 'retrieved all {0} Reuters url links'.format(len(reuters_all_urls))

    # slate politics and international news: 149, Feb 2001
    slate_urls = get_list_of_all_urls(slate_url_base,slate_klass,slate_klass2,range(60))
    slate_urls2 = get_list_of_all_urls(slate_url_base,'tiles','tile basic',range(60,150))

    all_slate_urls = slate_urls + slate_urls2

    extract_and_save_data('slate_data.json', get_slate_datapoint_info,slate_urls)
    '''
