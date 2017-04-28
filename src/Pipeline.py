def run_all_classifiers(data):

    X = data[data.author_gender != 'unknown']

    text = X['article']

    y = get_y_target_values(X, 'author_gender')
    X.drop(['author_gender', 'article', 'title', 'date_posted',
            'average_date', 'year', 'newssite'], axis=1, inplace=True)
    # X = X[['article_len','mean_sentence_len','mean_word_len',
    #        'type_token_ratio', 'freq_ifs', 'freq_quotation_marks',
    #        'freq_semi_colons', 'freq_verys', 'polarity',
    #        'std_sentence_len','subjectivity']]

    min_polarity = X['polarity'].min()
    X['polarity'] = X['polarity'] + (min_polarity * -1)

    X = X.reset_index()
    X.drop('author', axis=1, inplace=True)

    X = vectorize_articles(text, meta_data=X)
    X = X.toarray()

    clfs = [AdaBoostClassifier(n_estimators=100, learning_rate=0.05),
            RandomForestClassifier(n_estimators=100, max_depth=20,
                                   criterion='gini')]

    gender_model = Classifiers(clfs)
    gender_model.train(X, y)  # train on train data
    # gender_model.cross_validate(X,y)
    # gender_model.plot_roc_curve() # run on test data
    # gender_model.test()

if __name__ == "__main__":
    media_sites = ['Slate', 'BuzzFeed', 'TIME', 'Atlantic']

    media_data = []
    media_grouped_data = []
    for media_name in media_sites:
        data, grouped_data = get_df_and_grouped_df(media_name,
                                                   filter_date=True,
                                                   min_date=2011)
        media_data.append(data)
        media_grouped_data.append(grouped_dat