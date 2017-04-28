from __future__ import division
import scipy
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.charts import Bar
from itertools import izip
import Filtering
import Countries
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.cm as cm
from itertools import izip
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter

YLABELS = {'article_len': 'Article Length', 'polarity': 'Polarity',
           'subjectivity': 'Subjectivity',
           'type_token_ratio': 'Type Token Ratio',
           'mean_sentence_len': 'Mean Sentence Length',
           'mean_word_len': 'Mean Word Length',
           'std_sentence_len': 'Std of Sentence Length',
           'freq_exclamation_marks': "Frequency of '!'",
           'freq_question_marks': "Frequency of '?'",
           'freq_ids': "Frequency of 'if's",
           'freq_verys': "Frequency of 'very'"}


def print_gender_stats(df, female_df, male_df):
    for feature in df.columns:
        if isinstance(df[feature][0], int) or isinstance(df[feature][0], float):
            print '________________{}______________'.format(feature)
            print 'mean: {:.3f} +- {:.3f}'.format(df[feature].mean(), df[feature].std()/np.sqrt(len(df)))
            print '{} females, {} males'.format(len(female_df), len(male_df))
            female_mean= female_df[feature].mean()
            male_mean= male_df[feature].mean()

            # use scipy two-tailed tests for difference
            tstat, p_value = scipy.stats.ttest_ind(female_df[feature], male_df[feature],equal_var=False)

            if p_value < 0.05:
                if female_mean > male_mean:
                    print 'Female {} significantly larger with mean of {}'.format(feature, female_mean)
                elif male_mean > female_mean:
                    print 'Male {} significantly larger with mean of {}'.format(feature, male_mean)
            else:
                print '{}'.format(df[feature].mean())


def print_gender_percentage(grouped_data):
    genders = [get_author_gender(name.split()[0]) for name in grouped_data.index if len(name.split())>0]
    females = genders.count('female')
    males = genders.count('male')
    total = females + males

    print '{}: {} females, {} males'.format(media_name,females/total, males/total)


def label_fig_xaxis_date(feature, media_names):
    plt.xticks(rotation=60)
    plt.ylabel(YLABELS.get(feature, feature))
    plt.xlabel('Date of publication')
    plt.legend(media_names)


def label_fig_xaxis_names(names, width=0):
    x_range = np.arange(len(names))
    plt.xticks(x_range + width, names, rotation='vertical')
    plt.xlim([x_range.min()-0.5,x_range.max()+0.5 + width])


def create_errorbar_graph_of_mean_feature_values(means, errs, names, feature,
                                                 colors):
    for i, mean, err, color in izip(xrange(len(means)), means, errs, colors):
        plt.errorbar(i, mean, yerr=err, fmt='o', color=color, alpha=0.5)
    plt.ylabel(YLABELS.get(feature, feature))
    label_fig_xaxis_names(names)


def plot_feature_vs_date(media_data, media_names, filename, colors):
    plt.style.use('fivethirtyeight')
    for feature in media_data[0].columns:
        if isinstance(media_data[0].iloc[0][feature], int) or \
                isinstance(media_data[0].iloc[0][feature], float):
            means = []
            errs = []
            plt.figure(figsize=(10, 4))
            for newssite_data, color in izip(media_data, colors):
                plt.subplot(1, 2, 1)
                data_grouped_by_month = Filtering.group_by_month(newssite_data)

                plt.plot(data_grouped_by_month.index,
                         data_grouped_by_month[feature], 'o-', color=color,
                         alpha=0.5)

                means.append(newssite_data[feature].mean())
                errs.append(newssite_data[feature].std() / np.sqrt(len(newssite_data)))
            label_fig_xaxis_date(feature, media_names)
            plt.subplot(1, 2, 2)
            create_errorbar_graph_of_mean_feature_values(means, errs,
                                                         media_names, feature,
                                                         colors)
            plt.tight_layout()
            plt.savefig(filename.format(feature), facecolor='white')
            plt.close()


def create_interactive_bar_plot(x, y, newssite):

    output_file("top_country_bar_graphs.html")

    p1 = Bar(y, title='Top mentioned countries', width=600, height=400)
    tab = Panel(child=p1, title=newssite)

    return tab


def plot_top_countries_for_each_newssite(media_data, media_sites):
    bar_width = 0.35
    tabs = []

    colors = {'Israel': 'c', 'Iran': 'cadetblue', 'Russia': 'salmon',
              'Syria': 'indianred', 'Georgia': 'burlywood', 'India': 'y',
              'Iraq': 'slategrey', 'Cuba': 'darkseagreen', 'China': 'tan',
              'Ukraine': 'teal', 'North Korea': 'lightcoral',
              'Niger': 'lightgray', 'Malaysia': 'goldenrod',
              'Mexico': 'darkolivegreen'}

    n = len(media_data)/2  # number of subplots in row
    m = 2  # number of subplots in column

    h = 5 * n
    w = 8 * m

    plt.figure(figsize=(w, h))
    for i, data, name in izip(xrange(1, len(media_data)+1), media_data, media_sites):
        countries_to_plot, counts_to_plot = Countries.get_sorted_top_countries(data)
        plt.subplot(n, m, i)
        set_colors = [colors.get(country,'g') for country in countries_to_plot]
        plt.bar(xrange(7), counts_to_plot,bar_width,color=set_colors)
        plt.xticks(np.arange(7) + bar_width / 2, countries_to_plot, rotation='vertical')
        plt.title(name,fontsize=20)
        #plt.savefig('images/top_countries_{}.jpg'.format(media_sites[i]))

    plt.style.use('fivethirtyeight')
    plt.savefig('images/top_countries.jpg')
    plt.close()
    #tabs.append(create_interactive_bar_plot(counts_to_plot,countries_to_plot,media_sites[i]))

    #all_tabs = Tabs(tabs=tabs)
    #show(all_tabs)

def plot_features_by_gender(data,filename,colors):
    female_df, male_df = Filtering.seperate_by_gender(data)

    print 'number of data: {}'.format(len(male_df)+len(female_df))
    plot_feature_vs_date([female_df, male_df], ['female', 'male'], filename, colors)


def count_by_year(df):
    return df.groupby(['year']).count()


def plot_gender_vs_time(data):
    female_data, male_data = Filtering.seperate_by_gender(data)
    female_data = count_by_year(female_data)
    male_data = count_by_year(male_data)

    plt.plot(female_data.index,female_data['article_len'],'o-')
    plt.plot(male_data.index,male_data['article_len'],'o-')
    min_year = female_data.index.min()
    max_year = female_data.index.max()
    plt.xlim([min_year,max_year])

    plt.legend(['Female','Male'])

    plt.savefig('images/gender_by_time.jpg', facecolor='whitesmoke')
    plt.close()


def get_gender_percentages(data):
    unknown_data = data[data.author_gender == 'unknown']
    female_data,male_data = Filtering.seperate_by_gender(data)

    percent_unknown = len(unknown_data) / len(data)
    percent_female = len(female_data) / len(data)
    percent_male = len(male_data) / len(data)

    return percent_female, percent_male, percent_unknown


def plot_gender_by_newssite(media_data,media_names):
    for i,data,name in izip(xrange(len(media_data)),media_data,media_names):
        percent_female, percent_male, percent_unknown = get_gender_percentages(data)

        print 'number of authors in {} dataset: {}'.format(name,len(data))

        width = 0.35
        p1 = plt.bar(i, percent_male, width, color='indianred')
        p2 = plt.bar(i, percent_female, width,bottom=percent_male,color='dimgray')
        p3 = plt.bar(i, percent_unknown, width,bottom=percent_female+percent_male,color='burlywood')

    plt.ylabel('Percentage')
    label_fig_xaxis_names(media_names,width=width/2)
    plt.legend((p1[0], p2[0], p3[0]), ('Men', 'Women', 'Unknown'), loc="best")
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.savefig('gender_percentages.jpg',facecolor='white')
    plt.close()

def print_top_words_of_all_articles(articles):
    stop_words = stopwords.words('english')
    stop_words = stopwords + ['said','also','says','would']

    words = [word.strip(punctuation) for article in articles for word in article.lower().split() if word.strip(punctuation) not in stop_words]
    stop_words_in_article = [word.strip(punctuation) for article in articles for word in article.lower().split() if word.strip(punctuation) in stop_words]

    print Counter(words).most_common(15)
    print Counter(stop_words_in_article).most_common(15)

def plot_clf_scores(scores,names,plot_name):
    plt.bar(xrange(len(scores)),scores,width=0.35)
    x_range = np.arange(len(names))
    plt.xticks(x_range, names, rotation='vertical')
    plt.xlim([x_range.min()-0.5,x_range.max()+0.5])
    plt.ylabel('Mean F1 Score')
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.savefig(plot_name,facecolor='whitesmoke')
    plt.close()
