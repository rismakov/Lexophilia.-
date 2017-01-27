from __future__ import division
import unicodedata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from stop_words import get_stop_words
from itertools import izip
from string import punctuation
from stylometry_analysis import StyleFeatures
import matplotlib.pyplot as plt
import json
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import time
import seaborn as sbn
import scipy
import networkx as nx
from collections import defaultdict, Counter
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.charts import Bar
from Classifiers import Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
from Naive_Bayes_predictor import NaiveBayesPredictor
from sklearn.svm import SVC
#import gridspec

# HOW DO I SORT ABOVE???

def print_gender_stats(df,female_df, male_df, feature):
    print '________________{}______________'.format(feature)
    print 'mean: {:.3f} +- {:.3f}'.format(df[feature].mean(),df[feature].std()/np.sqrt(len(df)))
    print '{} females, {} males'.format(len(female_df),len(male_df))
    female_mean= female_df[feature].mean()
    male_mean= male_df[feature].mean()

    #use scipy two-tailed tests for difference
    tstat, p_value = scipy.stats.ttest_ind(female_df[feature],male_df[feature],equal_var=False)

    if p_value < 0.05:
        if female_mean > male_mean:
            print 'female larger. significant gender difference?: {}'.format(p_value)
        elif male_mean > female_mean:
            print 'male larger. significant gender difference?: {}'.format(p_value)

def add_midpoint_date_to_df(df, grouped_df):
    '''
    Adds the midpoint date of the grouped data to grouped dataframe.

    Input: dataframe, grouped dataframe
    Output: grouped dataframe with additional datetime column
    '''
    average_dates=[]
    for author in grouped_df.index:
        all_dates_of_author = df[df.author==author]['date_posted']
        average_dates.append(all_dates_of_author.min() + \
            (all_dates_of_author.max() - all_dates_of_author.min())/2)

    grouped_df['average_date'] = average_dates
    return grouped_df

def group_by_month(df):
    grouped_by_time = df.set_index('average_date')
    return grouped_by_time.resample('5m').mean()

def get_author_gender(name):
    '''
    Input: name of author
    Output: string "female", "male", or "unknown"
    '''
    with open('female_names.txt') as f:
        female_names = set(f.read().splitlines())
    with open('male_names.txt') as f:
        male_names = set(f.read().splitlines())

    if name in female_names:
        return 'female'
    elif name in male_names:
        return 'male'
    return 'unknown'

def print_gender_percentage(grouped_data):
    genders = [get_author_gender(name.split()[0]) for name in grouped_data.index if len(name.split())>0]
    females = genders.count('female')
    males = genders.count('male')
    total = females + males

    print '{}: {} females, {} males'.format(media_name,females/total, males/total)

def group_by_author(df, print_gender_info = False):
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
    grouped_df = pd.concat([mean_df,mode_df], axis=1)
    grouped_df = add_midpoint_date_to_df(df, grouped_df)

    grouped_df['year'] = [int(num) for num in grouped_df.year] # change "mean"-years to ints

    if print_gender_info:
        print_gender_percentage(grouped_df)

    return grouped_df

def add_title_countries_to_df(df):
    with open('countries.txt') as f:
        countries = f.readlines()

    df['countries_in_title']=[[country for country in countries if country in title] for title in df.title]
    return df

def filter_data(df, filter_date=False,min_year=None):
    '''

    '''

    if 'word_diversity' in df.columns: # same as 'type-token-ratio', can remove
        df.drop(['word_diversity'],axis=1,inplace=True)

    df['date_posted'] = df.date_posted.apply(lambda date: datetime.fromtimestamp(date/1000))
    df['year'] = [date.year for date in df.date_posted]
    if filter_date:
        df = df[df.year >= min_year]
    #df = add_title_countries_to_df(df)
    print '{} articles'.format(len(df))
    return df

def open_and_filter_data(newssite, filter_date= False,min_year= None):
    df = pd.read_json('feature_data/{}.json'.format(newssite))

    print '____________________{}____________________'.format(newssite)
    print '{} datapoints retreived'.format(len(df))

    df = filter_data(df,filter_date,min_year)
    return df

def get_df_and_grouped_df(newssite,filter_date= False,min_date= None):
    df = open_and_filter_data(newssite,filter_date,min_date)
    grouped_df = group_by_month(group_by_author(df))
    return df, grouped_df

def label_fig(feature,media_names):
    plt.ylabel('{}'.format(feature))
    plt.xlabel('Date of publication')
    plt.legend(media_names)

def create_errorbar_graph_of_mean_feature_values(means,errs,names,feature):
    x_range = np.arange(len(names))
    for i, mean in enumerate(means):
        plt.errorbar(i,mean, yerr = errs[i], fmt ='o')
    plt.ylabel(feature)
    plt.xticks(x_range/ 2, names, rotation='vertical')
    plt.xlim([x_range.min()-0.5,x_range.max()+0.5])

def plot_feature_vs_date(media_data,media_names,filename):
    for feature in media_data[0].columns:
        means=[]
        errs=[]
        #gs = gridspec.GridSpec(1,2,width_rations=[3,2])
        plt.figure(figsize=(10,4))
        for newssite_data in media_data:
            plt.subplot(1,2,1)
            plt.plot(newssite_data.index, newssite_data[feature], 'o-', alpha=0.5)
            means.append(newssite_data[feature].mean())
            errs.append(newssite_data[feature].std() / np.sqrt(len(newssite_data)))
        label_fig(feature,media_names)
        plt.subplot(1,2,2)
        create_errorbar_graph_of_mean_feature_values(means,errs,media_names,feature)

        plt.tight_layout()
        plt.savefig(filename.format(feature))
        plt.close()

def count_country_mentions(df):
    all_countries = [country for countries in df['countries_in_title'] for country in countries]

    headlines = [headline for i,headline in enumerate(df['title']) if len(df['countries_in_title'].iloc[i]) > 0]
    country_count=Counter(all_countries)
    return country_count

def get_sorted_top_countries(data):
    data = add_title_countries_to_df(data)
    country_count = count_country_mentions(data)
    list_counts = [(country, cnt) for country,cnt in country_count.iteritems()]
    list_counts.sort(key = lambda tup: tup[1])
    countries_to_plot = [tup[0] for tup in list_counts][-7:]
    counts_to_plot = [tup[1] for tup in list_counts][-7:]
    return countries_to_plot,counts_to_plot

def create_interactive_bar_plot(x,y,newssite):

    output_file("top_country_bar_graphs.html")

    p1 = Bar(y, title = 'Top mentioned countries',width=600, height=400)
    tab = Panel(child=p1, title=newssite)

    return tab

def plot_top_countries_for_each_newssite(media_data,media_sites):
    bar_width = 0.35
    tabs=[]

    colors = {'Israel': 'c', 'Iran': 'cadetblue', 'Russia':'salmon','Syria':'indianred', \
                'Georgia':'burlywood', 'India':'y','Iraq':'slategrey' ,'Cuba':'darkseagreen',\
                'China':'tan', 'Ukraine':'teal','North Korea':'lightcoral'}
    n=2 #number of subplots in row
    m=2 #number of subplots in column

    h= 5 * n
    w= 8 * m

    plt.figure(figsize=(w,h))
    for i,data in enumerate(media_data):
        countries_to_plot,counts_to_plot = get_sorted_top_countries(data)
        plt.subplot(n,m,i+1)
        set_colors = [colors[country] for country in countries_to_plot]
        plt.bar(range(7),counts_to_plot,bar_width,color=set_colors)
        plt.xticks(np.arange(7) + bar_width / 2, countries_to_plot, rotation='vertical')
        plt.title(media_sites[i],fontsize=20)
        #plt.savefig('images/top_countries_{}.jpg'.format(media_sites[i]))
    plt.savefig('images/top_countries.jpg',facecolor=(248,248,248))
    plt.close()
    #tabs.append(create_interactive_bar_plot(counts_to_plot,countries_to_plot,media_sites[i]))

    #all_tabs = Tabs(tabs=tabs)
    #show(all_tabs)

def get_words(inds,tfidf):
    words = [word for word,i in tfidf.vocabulary_.iteritems() if i in inds]
    return words

def get_top_words_from_kmeans(media_data,media_sites):
    for i, data in enumerate(media_data):
        word_mat = vectorize_articles(data['article'])

        kmeans = KMeans(n_clusters = 3, random_state=0).fit(word_mat)

        print 'Top 10 words for each cluster in {}:'.format(media_sites[i])
        for cluster in kmeans.cluster_centers_:
            top_inds = np.argsort(cluster)[-8:]
            top_words = get_words(top_inds,tfidf)

            print top_words
            print '_________________'

def vectorize_articles(text):
    stop_words = stopwords.words('english')
    tfidf = TfidfVectorizer(stop_words = stop_words, max_features = 10000)
    return tfidf.fit_transform(text)

def find_top_words_of_articles():
    pass

def combine_data(data):
    return pd.concat(data)

def seperate_by_gender(df):
    return df[df.author_gender == 'female'], df[df.author_gender == 'male']

def plot_features_by_gender(data,filename):
    female_df, male_df = seperate_by_gender(data)
    grouped_female_data = group_by_author(female_df)
    grouped_male_data = group_by_author(male_df)

    plot_feature_vs_date([grouped_female_data,grouped_male_data],['female','male'],filename)

def add_title_countries_to_df(df):
    with open('countries.txt') as f:
        countries = [country.strip('\n') for country in f.readlines()]

    df['countries_in_title']=[[country for country in countries if country in title] for title in df.title]
    return df

def remove_nodes(nodes_dict):
    keys_to_remove = []
    for k,v in nodes_dict.iteritems():
          if len(v) < 2:
              keys_to_remove.append(k)

    for key in set(keys_to_remove):
        del nodes_dict[key]
    return nodes_dict

def remove_weak_connections(nodes_dict,threshold):
    for k,v in nodes_dict.iteritems():
        nodes_dict[k] = [country for country in v if v.count(country) > threshold]

    #nodes_dict = {k: v for k, v in nodes_dict.iteritems() if v}
    return nodes_dict

def create_node_dict(df):
    nodes_dict = defaultdict(list)
    for items in df['countries_in_title']: #countries in every article
        if len(items)>1:
            for i, word in enumerate(items): # country for specific article
                nodes_dict[word] += (items[:i] + items[i+1:])

    #nodes_dict = remove_nodes(nodes_dict)
    nodes_dict = remove_weak_connections(nodes_dict,3)
    G=nx.Graph(nodes_dict)
    return G

def create_node_graph(df,graph_name):
    df= add_title_countries_to_df(df)
    nx.write_gml(create_node_dict(df), graph_name)

def filter_out_unknowns(data):
    return data[data.author_gender != 'unknown']

def count_by_month(df):
    grouped_by_time = df.set_index('average_date')
    grouped_by_time = grouped_by_time.resample('5m').count()
    return grouped_by_time

def plot_gender_vs_time(df):
    filtered_data = filter_out_unknowns(df) #57490 total articles
    data = group_by_author(filtered_data)
    female_data = data[data.author_gender == 'female']
    male_data = data[data.author_gender == 'male']

    females = female_data.groupby(['year']).count()['article']
    males = male_data.groupby(['year']).count()['article']

    gender_diff = (males-females)/(males+females)

    plt.plot(gender_diff.index,gender_diff,'o-')
    plt.xlim([datetime(2000,1,1),datetime(2017,1,1)])

    plt.savefig('images/gender_by_time.jpg',facecolor=(248,248,248))
    plt.close()

def get_y_target_values(data,target_feature):
    y = data.copy()
    y = y[target_feature].reset_index()
    y= y.replace(to_replace=['female','male'],value=[0,1])
    y= y[target_feature].astype(int)
    return np.array(y)

def run_all_classifiers(data):
    X = data[data.author_gender != 'unknown']
    X = group_by_author(X)

    y = get_y_target_values(X,'author_gender')
    X.drop(['author_gender','article','title','date_posted','average_date','freq_israels','freq_syrias','year'],axis=1, inplace=True)
    X = pd.get_dummies(X)
    X.drop('newssite_slate',axis=1,inplace=True)
    X['constant'] = 1

    gender_model = Classifiers([AdaBoostClassifier(),RandomForestClassifier(),DecisionTreeClassifier(), GradientBoostingClassifier(),LogisticRegression()])
    gender_model.train(X,y)
    #gender_model.cross_validate(X,y)
    gender_model.plot_roc_curve()

if __name__ == "__main__":
    media_sites = ['Slate','Buzzfeed','TIME','Atlantic']

    media_data = []
    media_grouped_data = []
    grouped_female_data = []
    grouped_male_data = []
    for media_name in media_sites:
        data,grouped_data = get_df_and_grouped_df(media_name,filter_date=False,min_date=None)
        media_data.append(data)
        media_grouped_data.append(grouped_data)

    combined_data = combine_data(media_data).reset_index()

    # slate has more males than females compared to the other newssite. remove to make sure effect is due to gender and not newssite
    #combined_data_no_slate = combine_data(media_data[1:])

    #plot_features_by_gender(combined_data,'images/{}_by_gender.jpg')
    #plot_features_by_gender(combined_data_no_slate,'images/{}_by_gender_no_slate.jpg')

    #plot_feature_vs_date(media_grouped_data, media_sites,'images/{}.jpg')
    #plot_top_countries_for_each_newssite(media_data, media_sites)
    #create_node_graph(combined_data,'countries_in_titles_connections.gml')

    #plot_gender_vs_time(combined_data)

    run_all_classifiers(combined_data)

    '''
    filtered_data = data[data.author_gender != 'unknown']
    X = filtered_data['article'].reset_index()
    X.drop(['index'],axis=1,inplace=True)
    y = get_y_target_values(filtered_data,'author_gender')
    gender_model = NaiveBayesPredictor()
    gender_model.train(X,y)
    #gender_model.cross_validate(y)
    gender_model.test_results(y, average= 'macro')
    #gender_model.test_results()

    X = combined_data['article'].reset_index()
    y = np.array(combined_data['newssite'])
    gender_model = NaiveBayesPredictor()
    gender_model.train(X,y)
    #gender_model.cross_validate(y, average= 'macro')
    gender_model.test_results(y, average='macro')
    #gender_model.test_results()
    '''



    plt.close('all')
