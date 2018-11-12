import tensorflow as tf
import numpy as np
import nltk.data
import math
import re
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from preprocess import loadDatasetGen
from word2vec_training import loadWord2VecConvertedFromGlove, loadWord2VecOnGoogleDataset
from tensorflow.contrib import rnn

# Parameters
learning_rate = 0.001
epochs = 100
batch_size = 32
hidden_nodes = 128
dropout = 0.2
split_size = 10
head_max = 20
body_max = 80

# Location where model will be saved.
model_path = "./models/independent_encoding.ckpt" 

tf.reset_default_graph()

"""
Creates an Dynamic RNN with a lstmunit as it's cell. 
"""


def lstm_encoder(input_to_encoder, lstm_units):
    with tf.variable_scope('lstm'):
        lstm_cell =  tf.nn.rnn_cell.LSTMCell(lstm_units, name="basic_lstm_cell")
        return tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell,
            cell_bw=lstm_cell,
            dtype=tf.float64,
            inputs=input_to_encoder)


def encode_body(input_to_encoder, lstm_units, ini_state_fw, ini_state_bw):
    #input_to_encoder = tf.placeholder(shape = (None, None,300), dtype=tf.float64, name='input_to_encoder');
    with tf.variable_scope('lstm1'):
        
        lstm_cell =  tf.nn.rnn_cell.LSTMCell(lstm_units, name="basic_lstm_cell")
        return tf.nn.bidirectional_dynamic_rnn(
                            cell_fw=lstm_cell,
                            cell_bw=lstm_cell,
                            dtype=tf.float64,    
                            inputs=input_to_encoder,
                            initial_state_fw = ini_state_fw,
                            initial_state_bw = ini_state_bw)

# Tensorflow Graph

# Input to the lstm encoder. 300 because of wordvectors size (gensim word2vec).
input_to_encoder = tf.placeholder(shape=(None, None, 300), dtype=tf.float64, name='input_to_encoder');

# Model to get encoding
encoded_variables = lstm_encoder(input_to_encoder, hidden_nodes)

# Encoding the body

# initial state for the forward pass
initial_c_fw=tf.placeholder(tf.float64, [None, 128],name='initial_c_fw')
initial_h_fw=tf.placeholder(tf.float64, [None, 128],name='initial_h_fw')
initial_state_fw = rnn.LSTMStateTuple(c=initial_c_fw, h=initial_h_fw)

# initial state for the backward pass
initial_c_bw = tf.placeholder(tf.float64, [None, 128], name='initial_c_bw')
initial_h_bw = tf.placeholder(tf.float64, [None, 128], name='initial_h_bw')
initial_state_bw = rnn.LSTMStateTuple(c=initial_c_bw, h=initial_h_bw)

encoded_bodies = encode_body(input_to_encoder,128,initial_state_fw, initial_state_bw)
    
# Inputs and correct outputs
x_head = tf.placeholder(shape=([None, hidden_nodes]), dtype=tf.float64, name='x_head')
x_body = tf.placeholder(shape=([None, hidden_nodes]), dtype=tf.float64, name='x_body')
y = tf.placeholder(shape=[None, 4], dtype=tf.float64, name='y')

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Weights
weights = {
    'W_head': tf.get_variable("W_head", shape=[hidden_nodes, hidden_nodes],
                              dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
    'W_body': tf.get_variable("W_body", shape=[hidden_nodes, hidden_nodes],
                              dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
    'W2': tf.get_variable("W2", shape=[hidden_nodes, 4],
                          dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
}

# Biases
biases = {
    'b1': tf.get_variable("b1", shape=[1, hidden_nodes],
                          dtype=tf.float64, initializer=tf.constant_initializer(0)),

    'b2': tf.get_variable("b2", shape=[1, 4],
                          dtype=tf.float64, initializer=tf.constant_initializer(0))
}


# Feedforward neural network model
def forward_propagation(X_head, X_body):
    hidden_layer_head = tf.matmul(X_head, weights['W_head'])
    hidden_layer_body = tf.matmul(X_body, weights['W_body'])
    hidden_layer = hidden_layer_head + hidden_layer_body
    hidden_layer = tf.nn.relu(hidden_layer)
    hidden_layer_dropout = tf.nn.dropout(hidden_layer, keep_prob=(1 - dropout))
    out_layer = tf.matmul(hidden_layer_dropout, weights['W2']) + biases['b2']

    return out_layer


# Construct the model
y_hat = forward_propagation(x_head, x_body)

# Cost and optimizer functions
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy function
correct_predicton = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicton, tf.float64))

# Initializing the variables
init = tf.global_variables_initializer()


def clean(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

print("Loading word2vec model...")
# word2vec_model = loadWord2VecConvertedFromGlove()
word2vec_model = loadWord2VecOnGoogleDataset()
print("Finished loading word2vec model.")

def prepare_dataset():
    #print("Loading word2vec model...")
    # word2vec_model = loadWord2VecConvertedFromGlove()
    #word2vec_model = loadWord2VecOnGoogleDataset()
    #print("Finished loading word2vec model.")

    print("Getting dataset...")
    for headline_body_pairs, stances in loadDatasetGen():    
        print("Finished getting dataset.")
    
        stance_labelencoder = LabelEncoder()
        stances_label_encoded = stance_labelencoder.fit_transform(stances)
        stances_label_encoded = stances_label_encoded.reshape(len(stances_label_encoded), 1)
        onehotencoder = OneHotEncoder(sparse=False)
        stances_onehotencoded = onehotencoder.fit_transform(stances_label_encoded)
    
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
    
            # Zero padding for headline
            zeropadded_headline_vec = np.zeros((head_max, len(headline_vec[0])))
            zeropadded_headline_vec[:headline_vec.shape[0], :headline_vec.shape[1]] = headline_vec
    
            # zero padding for body
            zeropadded_body_vec = np.zeros((body_max, len(body_vec[0])))
            zeropadded_body_vec[:body_vec.shape[0], :body_vec.shape[1]] = body_vec
    
            # concatenating the headline and body vectors
            headline_body_pairs_vec[i] = np.array(np.concatenate((zeropadded_headline_vec, zeropadded_body_vec), axis=0))
    
        print('Headline body pairs formed.')
        #del headline_body_pairs
        #del stances
        #del word2vec_model
        #gc.collect()
        yield headline_body_pairs_vec, stances_onehotencoded


def split_dataset(x, y, z):
    X_train, X_dev, Y_train, Y_dev, z_train, z_dev = train_test_split(x, y, z, test_size=0.1, random_state=42)
    return X_train, X_dev, Y_train, Y_dev, z_train, z_dev


def train(session, X_train_head, X_train_body, y_train):
    print("\n")
    total_batch = int(math.ceil(len(X_train_head) / batch_size))
    for epoch in range(epochs):
        avg_cost = 0
        loss = 0
        start, end = 0, batch_size
        for i in range(total_batch):
            batch_x_head = X_train_head[start:end]
            batch_x_body = X_train_body[start:end]
            batch_y = y_train[start:end]

            _, loss = session.run([optimizer, cost], feed_dict={x_head: batch_x_head, x_body: batch_x_body, y: batch_y})
            avg_cost += loss
            start += batch_size
            # if it is last batch then the (end) will be the length of the X_train
            # else shift by batch size.
            if i == total_batch - 2:
                end = len(X_train_head)
            else:
                end += batch_size
        avg_cost = avg_cost / total_batch
        train_accuracy = session.run(accuracy, feed_dict={x_head: X_train_head, x_body: X_train_body, y: y_train})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "accuracy =", "{:.3f}".format(train_accuracy))


def cross_validate(session, X_train_head, X_dev_head, X_train_body, X_dev_body, y_train, y_dev):
    results = []
    kf = KFold(n_splits=split_size)
    print('Cross validation .')
    for train_idx, val_idx in kf.split(X_train_head, X_train_body, y_train):
        # Training part
        train_x_head = X_train_head[train_idx]
        train_x_body = X_train_body[train_idx]
        train_y = y_train[train_idx]
        # Validation part
        val_x_head = X_train_head[val_idx]
        val_x_body = X_train_body[val_idx]
        val_y = y_train[val_idx]
        train(session, train_x_head, train_x_body, train_y)
        results.append(session.run(accuracy, feed_dict={x_head: val_x_head, x_body: val_x_body, y: val_y}))
    test_accuracy = session.run(accuracy, feed_dict={x_head: X_dev_head, x_body: X_dev_body, y: y_dev})
    return results, test_accuracy


def main():
    iter = 0
    with tf.Session() as session:
        
        for x, y in prepare_dataset():
                
        
            # Now process x
            # Input shape: x ---> (training samples, head_max+body_max, 300)
        
            headlines = x[:, :head_max, :]
            bodies = x[:, head_max:, :]
        
            del x
            gc.collect()
        
            # Configure GPU not to use all memory
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
    
            session.run(init)
            if iter != 0:
                saver.restore(session, model_path)
            outputs = session.run([encoded_variables], feed_dict={input_to_encoder: np.array(headlines)})
            
            output_fw, output_bw = outputs[0][0]
            states_fw, states_bw = outputs[0][1]
        
            state_c_fw = []
            state_c_bw = []
            state_h_fw = []
            state_h_bw = []
    
            for i in range(states_fw[0].shape[0]):
                state_c_fw.append(states_fw.c[i])
                state_h_fw.append(states_fw.h[i])
                state_c_bw.append(states_bw.c[i])
                state_h_bw.append(states_bw.h[i])
            
    
            del headlines
            del outputs
            gc.collect()
    
            outputs = session.run([encoded_bodies], feed_dict={input_to_encoder: np.array(bodies)
                                                        , initial_c_fw:np.array(state_c_fw)
                                                        , initial_c_bw:np.array(state_c_bw)
                                                        , initial_h_fw:np.array(state_h_fw)
                                                        , initial_h_bw:np.array(state_h_bw)});
            # outputs  = ppx[0][0]
            output_fw, output_bw = outputs[0][0]
            states_fw, states_bw = outputs[0][1]
            output_fw = np.transpose(output_fw, (1,0,2))
            output_bw = np.transpose(output_bw, (1,0,2))
    
            encodedd_op_batch_bodies1 = output_fw[-1]
            encodedd_op_batch_bodies2 = output_bw[-1]
            del bodies
            del outputs
            gc.collect()
    
            # At this encodedd_op_batch_XXX contains inputs
            X_train_head, X_dev_head, X_train_body, X_dev_body, y_train, y_dev = split_dataset(encodedd_op_batch_bodies1,
                                                                                               encodedd_op_batch_bodies2, y)
            # train(session, X_train, y_train)
            result, test_accuracy = cross_validate(session, X_train_head, X_dev_head, X_train_body, X_dev_body, y_train,
                                                   y_dev)
            print("\n")
            print("Cross-validation result: ", result)
            print("Training Accuracy: ",np.mean(np.array(result)))
            print("Test accuracy: ", test_accuracy)
            # Saving the model.
            save_path = saver.save(session, model_path)
            print("Model saved in file: %s" % save_path)
            
            iter= iter+1