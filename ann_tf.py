import tensorflow as tf
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from word2vec_training import loadWord2VecConvertedFromGlove, loadWord2VecOnGoogleDataset
from preprocess import processDataset

def prepareDataset():
    print("Loading word2vec model...")
    #word2vec_model = loadWord2VecConvertedFromGlove()
    word2vec_model = loadWord2VecOnGoogleDataset()
    print("Finished loading word2vec model.")
    print("Getting dataset...")
    headline_body_pairs, stances = processDataset("punctonly")
    print("Finished getting dataset.")
    
    stance_labelencoder = LabelEncoder()
    stances_label_encoded = stance_labelencoder.fit_transform(stances)
    stances_label_encoded = stances_label_encoded.reshape(len(stances_label_encoded), 1)
    onehotencoder = OneHotEncoder(sparse = False)
    stances_onehotencoded = onehotencoder.fit_transform(stances_label_encoded)
    
    headline_body_pairs_vec = np.zeros((len(headline_body_pairs), 600))
    
    for i in range(0, len(headline_body_pairs)):
        headline = headline_body_pairs[i][0]
        body = headline_body_pairs[i][1]

        headline_vec, body_vec = [], []
        for word in headline:
            if word in word2vec_model.vocab:
                word_vec = word2vec_model[word]
            else:
                word_vec = np.zeros((300,))
            headline_vec.append(np.array(word_vec))
            
        headline_vec = np.array(headline_vec)
        
        for word in body:
            if word in word2vec_model.vocab:
                word_vec = word2vec_model[word]
            else:
                word_vec = np.zeros((300,))
            body_vec.append(np.array(word_vec))
            
        body_vec = np.array(headline_vec)
        
        headline_vec_mean = np.mean(headline_vec, axis = 0)
        body_vec_mean = np.mean(body_vec, axis = 0)
        headline_body_vec_mean = np.concatenate((headline_vec_mean, body_vec_mean), axis = 0)
        
        for j in range(0, 600):
            headline_body_pairs_vec[i][j] = headline_body_vec_mean[j]
    
    print("Finished preparing dataset.")
    return headline_body_pairs_vec, stances_onehotencoded

# Split dataset into train and dev sets
def splitDataset():
    print("Splitting dataset...")
    X, y = prepareDataset()
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print("Finished splitting dataset.")
    return X_train, X_dev, y_train, y_dev

def buildModel(X_train, X_dev, y_train, y_dev):
    
    learning_rate = 0.001
    epochs = 10
    batch_size = 32
    
    x = tf.placeholder(tf.float32, [None, 600])
    y = tf.placeholder(tf.float32, [None, 4])
    
    W1 = tf.get_variable(name = "W1", shape = [600, 128], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([128], name = "b1"))
    
    W2 = tf.get_variable(name = "W2", shape = [128, 4], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([4], name = "b2"))
    
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)
    hidden_out = tf.nn.dropout(hidden_out, keep_prob = 0.8)
    
    y_ = tf.add(tf.matmul(hidden_out, W2), b2)
    
    #y_ = tf.nn.softmax(y_)
    #y_clipped = tf.clip_by_value(y_, 1e-10, 0.999999)
    #cross_entropy = tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis = 1))
    
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cross_entropy)
    
    init_op = tf.global_variables_initializer()
    
    correct_prediciton = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediciton, tf.float32))
    
    with tf.Session() as sess:
        
        sess.run(init_op)
        total_batch = int(math.ceil(len(X_train)/batch_size))
        for epoch in range(epochs):
            avg_cost = 0
            start, end = 0, 32
            for i in range(total_batch):
                start = start + (i*batch_size)
                if i == total_batch - 1:
                    end = len(X_train) - 1
                else:
                    end = end + (i*batch_size)
                
                batch_x = X_train[start:end]
                batch_y = y_train[start:end]
                
                _, c = sess.run([optimizer, cross_entropy], feed_dict = {x: batch_x, y: batch_y})
                print("Batch: ", (i + 1))
                print("c = ", c)
                avg_cost += c/total_batch
                print("Avg Cost =", avg_cost)
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print(sess.run(accuracy, feed_dict={x: X_dev, y: y_dev}))