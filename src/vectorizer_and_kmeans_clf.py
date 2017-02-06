import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import seaborn as sns 
import cPickle

def get_words(inds,tfidf):
    words = [word for word,i in tfidf.vocabulary_.iteritems() if i in inds]
    return words

def get_top_words_from_kmeans(media_data,media_sites):
    for i, data in enumerate(media_data):
        word_mat, tfidf = vectorize_articles(data['article'])

        kmeans = KMeans(n_clusters = 3, random_state=0).fit(word_mat)

        print 'Top 10 words for each cluster in {}:'.format(media_sites[i])
        for cluster in kmeans.cluster_centers_:
            top_inds = np.argsort(cluster)[-8:]
            top_words = get_words(top_inds,tfidf)

            print top_words
            print '_________________'

def add_meta_data_to_tfidf_mat(sparse_mat,meta_data):
    meta_data =  sparse.csr_matrix(meta_data)
    return sparse.hstack([sparse_mat,meta_data])

def vectorize_articles(text,meta_data=None):
    #stop_words = stopwords.words('english')
    tfidf = TfidfVectorizer(max_features = 5000)
    word_mat = tfidf.fit(text) 

    with open('TFIDF_fit', 'wb') as f:
        cPickle.dump(word_mat, f)

    word_mat = word_mat.transform(text)
    word_mat = add_meta_data_to_tfidf_mat(word_mat,meta_data)
    
    return word_mat

