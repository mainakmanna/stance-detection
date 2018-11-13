# Import relevant packages and modules
from csv import DictReader
from csv import DictWriter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Initialise global variables

stop_words = list(set(stopwords.words('english')))

def get_headbody_data(file_instances, file_bodies):
    """
    Gets headline, body , staances from given file path.

    Args:
        file_instances: csv file containing stance/body combined information.
        file_bodies: csv file containing body information.

    Returns:
        heads: dictionary containing (k,v)=>headline,id
        bodies: dictionary containing (k,v)=>BodyId,articleBody
        instances: ordered dict of headline,bodyid,stance
    """


    # Load data
    # instances
    """ list of ordereddicts
    headline -> text
    bodyid -> num
    stance -> related/...
    """
    instances = read(file_instances)
    ## list->ordereddict -> (bodyid->num),(articlebody->blob)
    bodies_read = read(file_bodies)
    heads = {}
    bodies = {}

    # Process instances
    for instance in instances:
        # if not in heads{} then add it to heads with value len(dict)
        # { headine: unique_num}
        if instance['Headline'] not in heads:
            head_id = len(heads)
            heads[instance['Headline']] = head_id
        # converting to int simply for the bodyid
        instance['Body ID'] = int(instance['Body ID'])

    # Process bodies : copy from bodies to self.bodies
    for body in bodies_read:
        bodies[int(body['Body ID'])] = body['articleBody']

    return heads, bodies, instances

def read(filename):

    """
    Read Fake News Challenge data from CSV file

    Args:
        filename: str, filename + extension

    Returns:
        rows: list, of dict per instance

    """

    # Initialise
    rows = []

    # Process file
    with open(filename, "r", encoding='utf-8') as table:
        r = DictReader(table)
        for line in r:
            rows.append(line)

    return rows


# Define relevant functions
def pipeline_train(train, test, lim_unigram):
    """

    Process train set, create relevant vectorizers

    Args:
        train: object containing train set data
        test:  object containing test set data
        lim_unigram: int, number of most frequent words to consider

    Returns:
        train_set: list, of numpy arrays
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    """

    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    body_ids = []
    id_ref = {}
    train_set = []
    
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    test_body_ids = []
    head_tfidf_track = {}
    body_tfidf_track = {}

    # Identify unique heads and bodies
    # XXX_track variables are used only for keeping track of
    # appeared heads and bodies
    # heads(bodies): stores the unique heads(bodies).

    for instance in train['instances']:
        heads_track[instance['Headline']] = 1
        bodies_track[instance['Body ID']] = 1
    
    heads = list(heads_track.keys())
    body_ids = list(bodies_track.keys())

    for bodyid in body_ids:
        bodies.append(train['bodies'][bodyid])

    for instance in test['instances']:
        test_heads_track[instance['Headline']] = 1
        test_bodies_track[instance['Body ID']] = 1

    test_heads = list(test_heads_track.keys())
    test_body_ids = list(test_bodies_track.keys())

    for test_bodyid in test_body_ids:
        test_bodies.append(test['bodies'][test_bodyid])

    # Create reference dictionary
    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words). \
        fit(heads + bodies + test_heads + test_bodies)  # Train and test sets

    # Process train set
    for instance in train['instances']:
        head = instance['Headline']
        body_id = instance['Body ID']
        head_tf = tfreq[id_ref[head]].reshape(1, -1)
        body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
        if head not in head_tfidf_track:
            head_tfidf = tfidf_vectorizer.transform([head]).toarray()
            head_tfidf_track[head] = head_tfidf
        else:
            head_tfidf = head_tfidf_track[head]
        if body_id not in body_tfidf_track:
            body_tfidf = tfidf_vectorizer.transform([train['bodies'][body_id]]).toarray()
            body_tfidf_track[body_id] = body_tfidf
        else:
            body_tfidf = body_tfidf_track[body_id]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        
        # np.c_: Translates slice objects to concatenation along the second axis.
        # In this case [[h1,h2,...,h5000]][[b1,b2,...,b5000]][[cos]] will become
        # [[h1,h2,...,h5000,b1,b2,...,b5000,cos]]. Squeezing it becomes
        # [h1,h2,...,h5000,b1,b2,...,b5000,cos]--->feat_vec
        
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        train_set.append(feat_vec)

    return train_set, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """

    Process test set

    Args:
        test: FNCData object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:
        test_set: list, of numpy arrays

    """

    # Initialise
    test_set = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    # Process test set
    for instance in test['instances']:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            head_bow = bow_vectorizer.transform([head]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
            heads_track[head] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[head][0]
            head_tfidf = heads_track[head][1]
        if body_id not in bodies_track:
            body_bow = bow_vectorizer.transform([test['bodies'][body_id]]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([test['bodies'][body_id]]).toarray().reshape(1, -1)
            bodies_track[body_id] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body_id][0]
            body_tfidf = bodies_track[body_id][1]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        test_set.append(feat_vec)

    return test_set
