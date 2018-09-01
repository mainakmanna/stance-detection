import nltk.data
from gensim.models.keyedvectors import KeyedVectors
from preprocess import *
import numpy as np
import re
import gc

import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

n_max=20
m_max=80
w = np.zeros( shape= ( 49972, 100, 300))

#word_vectors = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True)

def clean(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def get_data():
    headline_body_pairs, stances = loadDataset();
    #w = np.zeros( shape= ( len(headline_body_pairs), 100, 300))
    results = {'agree':0,'disagree':1,'discuss':2,'unrelated':3}
    p = [None] * len(headline_body_pairs)


    for i in range(0,  len(headline_body_pairs)):
        # Get the headline and body separately
        headline = headline_body_pairs[i][0]
        body = headline_body_pairs[i][1]

        stance = stances[i]
        stance = results[stance]
        p[i] = stance

        headline=clean(headline)
        body=clean(body)
        
         # tokenization of headline+body    
        headlinevec = nltk.word_tokenize(headline)
        bodyvec = nltk.word_tokenize(body)
        
        # avoiding the numbers
        hlv = np.array([np.array(word_vectors[w]) if w in word_vectors.vocab else np.zeros((300,)) for w in headlinevec])
        bv = np.array([np.array(word_vectors[w]) if w in word_vectors.vocab else np.zeros((300,)) for w in bodyvec])
        
        # Limiting the length upto n_max and m_max respectively.
        hlv = hlv[:n_max]
        bv = bv[:m_max]
        
        # Zero padding for headline
        zeropadded_headline_vec = np.zeros((n_max , len(hlv[0])))
        zeropadded_headline_vec[ :hlv.shape[0] , :hlv.shape[1]] = hlv

        # zero padding for body
        zeropadded_body_vec = np.zeros((m_max, len(bv[0])))
        zeropadded_body_vec[:bv.shape[0], :bv.shape[1]] = bv

        #print(zeropadded_headline_vec.shape)
        #print(zeropadded_body_vec.shape)
        
        # Concatenating them : ( 20 : 80 )
        w[i] = np.array(np.concatenate((zeropadded_headline_vec,zeropadded_body_vec),axis=0))
        print(i)
    #w = np.array(w)
    p = np.eye(len(results))[p]

    return w,p

    
def forward_propagation(ww,W1,b1,keep_prob,W2,b2):
    h = tf.nn.relu(tf.matmul(ww, W1) + b1)
    h_drop = tf.nn.dropout(h,keep_prob)
    yhat = tf.matmul(h_drop, W2) + b2
    return yhat

def plotGraph(epochList, lossList, trainAccList, testAccList):
    plt.figure(1)
    plt.plot(epochList, trainAccList, 'r--', label='Train Accuracy')
    plt.plot(epochList,testAccList,'g^',label='Test Accuracy' )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure(2)
    plt.plot(epochList,lossList,'bs',label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def encode_headline( input_to_encoder,lstmunit):
    lstm_cell =  rnn.BasicLSTMCell(lstmunit)
    return tf.nn.dynamic_rnn(
                            cell=lstm_cell,
                            dtype=tf.float64,    
                            inputs=input_to_encoder)

def encode_body( lstmunit):
    input_to_encoder = tf.placeholder(shape = (None, None,300), dtype=tf.float64, name='input_to_encoder');
    lstm_cell =  rnn.BasicLSTMCell(lstmunit)
    return tf.nn.dynamic_rnn(
                            cell=lstm_cell,
                            dtype=tf.float64,    
                            inputs=input_to_encoder)
    
    
def main():
    print('starting...')
    w,p = get_data()
    print('Got data...')
    
    
    # Now process w
    headlines = w[:, :n_max, :]
    bodies = w[:, n_max:, :]
    del w
    gc.collect()
    tf.reset_default_graph()
    
    input_to_encoder = tf.placeholder(shape = (None, None,300), dtype=tf.float64, name='input_to_encoder');
    encoded_headlines = encode_headline(input_to_encoder, 128)
    #encoded_bodies = encode_body(128)

    
    init = tf.global_variables_initializer()
    
    # Configure GPU not to use all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Start a new tensorflow session and initialize variables
    sess = tf.InteractiveSession(config=config)
    sess.run(init)
    
    ppx = sess.run([encoded_headlines], feed_dict={input_to_encoder: np.array(headlines)});
    outputs  = ppx[0][0]
    outputs  = np.transpose(outputs,(1,0,2))
    encodedd_op_batch_headlines = outputs[-1]
    del headlines
    gc.collect()
    ppx = sess.run([encoded_headlines], feed_dict={input_to_encoder: np.array(bodies)});
    outputs  = ppx[0][0]
    outputs  = np.transpose(outputs,(1,0,2))
    encodedd_op_batch_bodies = outputs[-1]
    del bodies
    gc.collect()
    sess.close()
    
#if __name__ == '__main__':
#    main()