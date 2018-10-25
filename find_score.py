import tensorflow as tf
import numpy as np
import nltk.data
import re
import gc
from preprocess import loadTestDataset
from word2vec_training import loadWord2VecConvertedFromGlove, loadWord2VecOnGoogleDataset

head_max = 20
body_max = 80

def clean(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def prepare_test_dataset():
    print("Loading word2vec model...")
    # word2vec_model = loadWord2VecConvertedFromGlove()
    word2vec_model = loadWord2VecOnGoogleDataset()
    print("Finished loading word2vec model.")

    print("Getting dataset...")
    headline_body_pairs = loadTestDataset();
    print("Finished getting dataset.")

    headline_body_pairs_vec = np.zeros(shape=(len(headline_body_pairs), 100, 300))

    for i in range(0, len(headline_body_pairs)):
        # getting the headline and body separately
        headline = headline_body_pairs[i][0]
        body = headline_body_pairs[i][1]

        # cleaning the headline and body
        headline = clean(headline)
        body = clean(body)

        # tokenizing of headline and body
        headline = nltk.word_tokenize(headline)
        body = nltk.word_tokenize(body)

        # getting word vectors, replacing unknown words and numbers with zero vectors
        headline_vec = np.array(
            [np.array(word2vec_model[word]) if word in word2vec_model.vocab else np.zeros((300,)) for word in headline])
        body_vec = np.array(
            [np.array(word2vec_model[word]) if word in word2vec_model.vocab else np.zeros((300,)) for word in body])

        # Limiting the length upto head_max and body_max respectively.
        headline_vec = headline_vec[:head_max]
        body_vec = body_vec[:body_max]

        # zero padding for headline
        zeropadded_headline_vec = np.zeros((head_max, len(headline_vec[0])))
        zeropadded_headline_vec[:headline_vec.shape[0], :headline_vec.shape[1]] = headline_vec

        # zero padding for body
        zeropadded_body_vec = np.zeros((body_max, len(body_vec[0])))
        zeropadded_body_vec[:body_vec.shape[0], :body_vec.shape[1]] = body_vec

        # concatenating the headline and body vectors
        headline_body_pairs_vec[i] = np.array(np.concatenate((zeropadded_headline_vec, zeropadded_body_vec), axis=0))
        
    print('Headline body pairs formed.')
    del headline_body_pairs
    del word2vec_model
    gc.collect()
    return headline_body_pairs_vec

def baseline_score():
    x = prepare_test_dataset()
    
    with tf.Session() as session:
        new_saver = tf.train.import_meta_graph('')
        new_saver.restore(session, "./models/baseline.ckpt")
        predictions = session.run(y_hat, feed_dict = {x: x})
        
