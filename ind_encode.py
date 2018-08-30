import nltk.data
from gensim.models.keyedvectors import KeyedVectors
from preprocess import *
import numpy as np
import re
import tensorflow as tf  
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

#word_vectors = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True)
n_max=20
m_max=80
def clean(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def get_data():
    headline_body_pairs, stances = loadDataset();
    w = [None] * 20
    results = {'agree':0,'disagree':1,'discuss':2,'unrelated':3}
    p = [None] * 20
    hll = [None] * 20
    bll = [None]*20
    for i in range(0,  20):
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
        
        hlv = hlv[:n_max]
        bv = bv[:m_max]

        zeropadded_headline_vec = np.zeros((n_max , len(hlv[0])))
        zeropadded_headline_vec[ :hlv.shape[0] , :hlv.shape[1]] = hlv

        zeropadded_body_vec = np.zeros((m_max, len(bv[0])))
        zeropadded_body_vec[:bv.shape[0], :bv.shape[1]] = bv
        
        hll[i] = zeropadded_headline_vec
        bll[i] = zeropadded_body_vec
        # columnwise me       
        #w[i] = np.concatenate((numpy_headlinevec,numpy_bodyvec),axis=0)
    p = np.eye(4)[p]
    
    return np.array(hll),np.array(bll),np.array(p)

    
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

def trythis(hl):
    lstm_cell =  rnn.BasicLSTMCell(128)
    outputs, last_states = tf.nn.dynamic_rnn(
                            cell=lstm_cell,
                            dtype=tf.float64,
                            sequence_length=20,
                            inputs=hl[:20])
    
    return outputs, last_states
        

    
def main():
    print('starting...')
    hl,bl, p = get_data()
    print('Got data...')
    tf.reset_default_graph()
    X = tf.placeholder(shape = (20,20,300), dtype=tf.float64, name='X');
    lstm_cell =  rnn.BasicLSTMCell(128)
    ppp = tf.nn.dynamic_rnn(
                            cell=lstm_cell,
                            dtype=tf.float64,    
                            inputs=X)
    
    ######################################
    
    # Create operation which will initialize all variables
    init = tf.global_variables_initializer()
    
    # Configure GPU not to use all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Start a new tensorflow session and initialize variables
    sess = tf.InteractiveSession(config=config)
    sess.run(init)
     
    ppx = sess.run([ppp], feed_dict={X: np.array(hl[:20])});
    outputs  = ppx[0][0]
    outputs  = np.transpose(outputs,(1,0,2))
    encodedd_op_batch = outputs[-1]
    tf.Print(tf.shape(ppx[0]));
    sess.close()
    