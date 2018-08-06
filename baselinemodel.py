import nltk.data
from gensim.models.keyedvectors import KeyedVectors
from preprocess import *
import numpy as np
import re
import tensorflow as tf  
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

word_vectors = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True)

def clean(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def get_data():
    headline_body_pairs, stances = loadDataset();
    w = [None] * len(headline_body_pairs)
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
        
        # columnwise mean
        numpy_headlinevec = np.mean(np.array(hlv),axis=0)
        numpy_bodyvec = np.mean(np.array(bv),axis=0)
        
        w[i] = np.concatenate((numpy_headlinevec,numpy_bodyvec),axis=0)
        
    w = np.array(w)
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

def main():
    print('starting...')
    w,p = get_data()
    print('Got data...')
    train_X, test_X, train_y, test_y = train_test_split(w, p, test_size=0.33, random_state=42)
    # Create the tensorflow model
    
    print('Building model...')
    tf.reset_default_graph()
    
    hidden_nodes = 128
    keep_prob = tf.placeholder(tf.float64)
    
    # ww is the input matrix     
    ww = tf.placeholder(shape=(None,w[0].shape[0]), dtype=tf.float64, name='ww')
    
    # pp is the correct labels (stances)
    pp = tf.placeholder(shape=(None, 4), dtype=tf.float64, name='pp')
    
    W1 = tf.get_variable("W1", shape=[w[0].shape[0], hidden_nodes],
               dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", shape=[hidden_nodes, 4],
               dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())
    
    b1 = tf.get_variable("b1", shape=[1,hidden_nodes], dtype=tf.float64,
                        initializer=tf.constant_initializer(0))
    b2 = tf.get_variable("b2", shape=[1,4], dtype=tf.float64,
                        initializer=tf.constant_initializer(0))
    
    # Forward propagation
    yhat = forward_propagation(ww,W1,b1,keep_prob,W2,b2)
    predict = tf.argmax(yhat,axis=1)
    predict = tf.cast(predict, tf.float64)
    
    # Backward Propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pp, logits=yhat))
    updates  = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    ######################################
    
    # Create operation which will initialize all variables
    init = tf.global_variables_initializer()
    
    # Configure GPU not to use all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Start a new tensorflow session and initialize variables
    sess = tf.InteractiveSession(config=config)
    sess.run(init)
    
    # These lists are for storing data later used in plotting
    epochList = []
    lossList = []
    trainAccList = []
    testAccList = []
    for epoch in range(10):
        avg_cost = 0.0
        loss = 0
        for i in range(len(train_X)):
            _,loss = sess.run([updates,cost], feed_dict={ww: np.matrix(train_X[i]), pp: np.matrix(train_y[i]),keep_prob : 0.2});
            avg_cost += loss
        avg_cost = avg_cost /  len(train_X)
        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={ww: train_X, pp: train_y,keep_prob:0.8}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={ww: test_X, pp: test_y,keep_prob:0.8}))
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%% , loss = %.2f%% "
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy, avg_cost*100))
        
        epochList.append(epoch+1)
        lossList.append(avg_cost*100.)
        trainAccList.append(train_accuracy*100.)
        testAccList.append(test_accuracy*100.)

    plotGraph(epochList, lossList, trainAccList, testAccList)
    
    sess.close()
    
if __name__ == '__main__':
    main()