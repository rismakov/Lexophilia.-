from __future__ import division
import requests
import unicodedata
from bs4 import BeautifulSoup
from string import punctuation
import json
import pickle


def convert_unicode_to_str(unicode):
    return unicodedata.normalize('NFKD', unicode).encode('ascii', 'ignore')

with open('text_files/female_names.txt') as f:
    female_names = set(f.read().splitlines())
with open('text_files/male_names.txt') as f:
    male_names = set(f.read().splitlines())


def ignore_words(string, words):
    for word in words:
        string = string.replace(word, '')
    return string


def get_soup(url):
    r = requests.get(url)
    return BeautifulSoup(r.text, 'html.parser')


def get_author_gender(name):
    if name in female_names:
        return 'female'
    elif name in male_names:
        return 'male'
    return 'unknown'


def get_h1_title(code_section):
    return code_section.h1.text


def get_date(code_section, klass):
    return code_section.find_all(class_=klass)[0].text


def get_author_name(code_section, klass):
    return code_section.find_all(class_='byline')[0].text


def get_unfiltered_code(code_section, klass):
    return code_section.find_all(class_=klass)


def get_breitbart_datapoint_info(urls, filename, datapoints=[]):
    ind = 0

    for i, url in enumerate(urls[ind:]):
        print url
        if url[:25] == 'http://www.breitbart.com/':
            if url[25:29] != 'live' and url[25:29] != 'tech' and url[25:30] != 'video'\
                  and url[25:30] != 'sport' and url[25:30] != 'big-h' and url[25:30] != 'big-j':
                print i

                datapoint = {}
                soup = get_soup(url)
                code_section = soup.article

                datapoint['title'] = get_h1_title(code_section)
                datapoint['date_posted'] = convert_unicode_to_str(get_date(code_section, 'bydate'))
                author_name = get_author_name(code_section, 'byauthor')
                datapoint['author'] = ' '.join(author_name.split()[1:])
                datapoint['author_gender'] = get_author_gender(author_name.split()[0])
                code_text = get_unfiltered_code(code_section, 'entry-content')

                code_text = code_text[0].text
                code_text = convert_unicode_to_str(code_text)

                datapoint['article'] = code_text
                datapoint['newssite'] = 'breitbart'
                datapoints.append(datapoint)

                save_to_json(filename, datapoints)

    return datapoints


def get_reuters_datapoint_info(urls, filename, datapoints):

    for i, url in enumerate(urls):

        if url[:18] == 'http://www.reuters':
            print i
            print url

            datapoint = {}
            soup = get_soup(url)
            code_section = soup.article

            try:

                try:
                    datapoint['title'] = get_h1_title(code_section)
                except:
                    datapoint['title'] = code_section.h2.text

                datapoint['date_posted'] = convert_unicode_to_str(get_date(code_section, 'time'))  # timestamp
                author_name = get_author_name(code_section, 'byline')  # author
                datapoint['author'] = author_name[1:]
                datapoint['author_gender'] = get_author_gender(author_name.split()[0])
                code_text = get_unfiltered_code(code_section, 'row col-md-9 col-md-offset-1 article-body-content') #article-text

                # if code_text:
                code_text = code_text[0].text
                code_text = convert_unicode_to_str(code_text)

                datapoint['article'] = code_text
                datapoint['newssite'] = 'reuters'
                datapoints.append(datapoint)

                save_to_json(filename, datapoints)

            except:
                pass

    return datapoints


def get_time_op_datapoint_info(urls, filename, datapoints=[]):

    ind = urls.index('http://time.com/90954/mothers-day-parenting-skills/')
    # ind=0

    for i, url in enumerate(urls[ind+1:]):
        print i, url

        datapoint = {}
        soup = get_soup(url)

        code_section = soup
        datapoint['title'] = soup.h1.text
        datapoint['date_posted'] = code_section.find_all('span', attrs={'class': 'MblGHNMJ'})[0].text
        author_name = code_section.find_all(class_='zhtAwgU0')[0].text
        datapoint['author'] = author_name
        datapoint['author_gender'] = get_author_gender(author_name.split()[0])

        code_text = get_unfiltered_code(code_section, 'column small-12 medium-10 medium-offset-1 large-offset-2  _10M0Ygc4')

        article = ''
        for code in code_text:
            article += code.text

        if isinstance(article, unicode):
            article = convert_unicode_to_str(article)

        datapoint['article'] = article
        datapoint['newssite'] = 'time opinion'
        datapoints.append(datapoint)

        save_to_json(filename, datapoints)

    return datapoints


def get_timemag_datapoint_info(urls, filename, datapoints=[]):

    # ind = urls.index('http://time.com/2819856/princess-letizia-spain-photos/')
    ind = 0

    for i, url in enumerate(urls[ind:]):
        print i
        print url

        datapoint = {}
        soup = get_soup(url)
        code_section = soup.article

        datapoint['title'] = get_h1_title(code_section)
        datapoint['date_posted'] = convert_unicode_to_str(get_date(code_section, 'publish-date'))
        author_name = get_author_name(code_section, 'byline')
        datapoint['author'] = author_name
        datapoint['author_gender'] = get_author_gender(author_name.split()[0])

        code_text = get_unfiltered_code(code_section, 'article-body')

        code_text = code_text[0].text
        code_text = convert_unicode_to_str(code_text)

        datapoint['article'] = code_text
        datapoint['newssite'] = 'time'
        datapoints.append(datapoint)

        save_to_json(filename, datapoints)

    return datapoints


def get_atlantic_datapoint_info(urls,filename,datapoints=[]):

    ind = urls.index('/international/archive/2011/12/dont-worry-north-korean-missiles-tests-are-only-routine/334125/')
    #ind = 0
    for i, url_end in enumerate(urls[ind+1:]):

        #get only urls from politics or international:
        if url_end[:9] == '/politics' or url_end[:14] == '/international':

            print i
            print url_end

            datapoint = {}
            url = 'https://www.theatlantic.com{}'.format(url_end)
            soup = get_soup(url)
            code_section =  soup.article

            datapoint['title'] = get_h1_title(code_section)
            datapoint['date_posted'] = convert_unicode_to_str(get_date(code_section,'date'))
            author_name = get_author_name(code_section,'byline')
            datapoint['author'] = author_name
            datapoint['author_gender'] = get_author_gender(author_name.split()[0])
            code_text = get_unfiltered_code(code_section,'article-body')

            code_text = code_text[0].text
            code_text = convert_unicode_to_str(code_text)

            datapoint['article'] = code_text
            datapoint['newssite'] = 'atlantic'
            datapoints.append(datapoint)

            save_to_json(filename, datapoints)

    return datapoints


def get_slate_datapoint_info(urls,file_name,datapoints):

    ind = urls.index('http://www.slate.com/articles/news_and_politics/politics/2007/07/bushs_latest_lame_libby_excuse.html')

    for i, url in enumerate(urls[ind+1:]):
        if url[:29] == 'http://www.slate.com/articles':
            print i
            print url

            datapoint = {}
            soup = get_soup(url)
            code_section = soup.find_all('article', class_='main')[0]

            datapoint['title'] = get_h1_title(code_section)
            datapoint['date_posted'] = convert_unicode_to_str(get_date(code_section, 'pub-date'))

            try:
                author_name = code_section.find_all('a', rel='author')[0].text
                datapoint['author'] = author_name

                datapoint['author_gender'] = get_author_gender(author_name.split()[0])
                code_text = get_unfiltered_code(code_section, 'newbody body parsys')

                if code_text:
                    code_text = code_text[0].text
                    code_text = convert_unicode_to_str(code_text)
                    code_text = ignore_words(code_text,['figure','homepage', 'width', 'height', 'url', 'background','bcvideo','liveeventrecirc','recirc','jpg','bp','videoid_', 'experienceid','weekof','brightcove','var','attr','setsize','taps','dam', 'tap'])

                    datapoint['article'] = code_text
                    datapoint['newssite'] = 'slate'
                    datapoints.append(datapoint)

                    with open(file_name, 'w') as outfile:
                        json.dump(datapoints, outfile)

            except:
                pass

    return datapoints


def get_bf_datapoint_info(urls,filename,datapoints):
    klass = 'c bf_dom'
    klass_alt = 'bf_dom c'

    ignore_inds=[]
    ignore_inds.append(urls.index('/jakel11/who-cared-the-most-about-the-government-shutdown'))
    ignore_inds.append(urls.index('/jeremybender/pro-tips-for-a-fantastic-filibuster'))
    ignore_inds.append(urls.index('/johnekdahl/14-principled-anti-war-celebrities-we-fear-may-hav-a1x1'))
    ignore_inds.append(urls.index('/senatorfranken/al-frankenas-13-reasons-why-the-minnesota-state-e04l'))
    ignore_inds.append(urls.index('/elongreen/mitts-reading-list-4yyh'))

    for i, url_end in enumerate(urls[ignore_inds[-1]+1:]):
        print i
        print url_end
        datapoint = {}
        url= 'https://www.buzzfeed.com{0}'.format(url_end)
        soup = get_soup(url)
        code_sections =  soup.find_all(class_=klass)
        if code_sections:
            code_section = code_sections[0]
        else:
            code_section =  soup.find_all(class_=klass_alt)[0]

        datapoint['title'] = get_h1_title(code_section)
        datapoint['date_posted'] = convert_unicode_to_str(get_date(code_section, 'post-datetimes'))
        author_name = get_author_name(code_section,'byline__author')
        datapoint['author'] = author_name

        if author_name == u'\n\n':
            datapoint['author_gender'] = 'unknown'
        else:
            datapoint['author_gender'] = get_author_gender(author_name.split()[0])

        code_text = get_unfiltered_code(code_section,'buzz_superlist_item_text')

        if code_text:
            code_text = code_text[0].text
            code_text = convert_unicode_to_str(code_text)
            code_text = code_text.replace('View this image','')
            code_text = code_text.replace(punctuation,'')

            articles.append(code_text)
            datapoint['article'] = code_text
            datapoint['newssite'] = 'buzzfeed'
            datapoints.append(datapoint)

            save_to_json(filename, datapoints)

    return datapoints


def save_to_json(filename, datapoints):
    with open(filename, 'w') as outfile:
        json.dump(datapoints, outfile)


def extract_and_save_data(name,scrape_func,urls,datapoints=[]):
    print 'retrieved all {0} url links'.format(len(urls))
    #datapoints, articles = call_func(scrape_func,urls)
    datapoints = scrape_func(urls,name,datapoints)
    save_to_json(name, datapoints)
    print 'retrieved all code'


def open_urls(file_name):
    with open(file_name, 'rb') as outfile:
        return pickle.load(outfile)


def open_json_file(filename):
    json_data = open(filename).read()
    return json.loads(json_data)

if __name__ == '__main__':

    breitbart_urls = open_urls('breitbart_urls')
    datapoints= open_json_file('data/breitbart_data.json')
    # datapoints=[]
    print 'retrieved {} datapoints'.format(len(datapoints))
    extract_and_save_data('data/breitbart_data2.json',get_breitbart_datapoint_info,breitbart_urls,datapoints)

    '''
    # time opinion
    time_urls = open_urls('time_opinion_urls')
    datapoints = open_json_file('data/time_opinion_data.json')
    print 'retrieved {0} datapoints'.format(len(datapoints))
    extract_and_save_data('data/time_opinion_data.json',get_time_op_datapoint_info,time_urls,datapoints)
    print 'retrieved all Time Opinion code'
    '''

    '''
    bf_urls = open_urls('buzzfeed_urls')
    datapoints= open_json_file('buzzfeed_data.json')
    print 'retrieved {0} datapoints'.format(len(datapoints))
    extract_and_save_data('buzzfeed_data.json',get_bf_datapoint_info,bf_urls,datapoints)
    '''
    '''
    # time world
    time_urls = open_urls('time_opinion_urls')
    #datapoints = open_json_file('data/time_opinion_data.json')
    #print 'retrieved {0} datapoints'.format(len(datapoints))
    extract_and_save_data('data/time_opinion_data.json',get_timemag_datapoint_info,time_urls,[])
    print 'retrieved all Time code'
    '''

    '''
    # reuters investigates- special reports
    reuters_urls = open_urls('urls/reuters_full_urls')
    extract_and_save_data('data/reuters.json',get_reuters_datapoint_info,reuters_urls,[])
    '''

    '''
    #the atlantic news
    atlantic_urls = open_urls('urls/atlantic_urls')
    datapoints = open_json_file('data/atlantic_data.json')
    print 'retrieved {0} datapoints'.format(len(datapoints))
    extract_and_save_data('data/atlantic_data.json',get_atlantic_datapoint_info,atlantic_urls,datapoints)
    print 'retrieved all Atlantic code'
    '''

    '''
    # slate politics and international news: 149, Feb 2001
    slate_urls = open_urls('slate_urls_list')
    datapoints= open_json_file('slate_data.json')
    print 'retrieved {0} datapoints'.format(len(datapoints))
    extract_and_save_data('slate_data.json',get_slate_datapoint_info,slate_urls,datapoints)
    '''
