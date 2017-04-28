from collections import Counter


def add_title_countries_to_df(df):
    with open('text_files/countries.txt') as f:
        countries = [country.strip('\n') for country in f.readlines()]

    df['countries_in_title'] = [[country for country in countries
                                 if country in title] for title in df.title]
    return df


def count_country_mentions(df):
    all_countries = [country for countries in df['countries_in_title']
                     for country in countries]

    headlines = [headline for i, headline in enumerate(df['title'])
                 if len(df['countries_in_title'].iloc[i]) > 0]
    country_count = Counter(all_countries)
    return country_count


def get_sorted_top_countries(data):
    data = add_title_countries_to_df(data)
    country_count = count_country_mentions(data)
    list_counts = [(country, cnt) for country, cnt in country_count.iteritems()]
    list_counts.sort(key=lambda tup: tup[1])
    countries_to_plot = [tup[0] for tup in list_counts][-7:]
    counts_to_plot = [tup[1] for tup in list_counts][-7:]
    return countries_to_plot, counts_to_plot
