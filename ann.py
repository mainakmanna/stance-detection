import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from word2vec_training import loadWord2VecConvertedFromGlove, loadWord2VecOnGoogleDataset
from preprocess import processDataset

def prepareDataset():
    #word2vec_model = loadWord2VecConvertedFromGlove()
    word2vec_model = loadWord2VecOnGoogleDataset()
    headline_body_pairs, stances = processDataset()
    
    stance_labelencoder = LabelEncoder()
    stances_label_encoded = stance_labelencoder.fit_transform(stances)
    stances_label_encoded = stances_label_encoded.reshape(len(stances_label_encoded), 1)
    onehotencoder = OneHotEncoder(sparse = False)
    stances_onehotencoded = onehotencoder.fit_transform(stances_label_encoded)
    
    headline_body_pairs_vec = [None] * len(headline_body_pairs)
    
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
        
        headline_body_pairs_vec[i] = np.concatenate((headline_vec_mean, body_vec_mean), axis = 0)
        
    return headline_body_pairs_vec, stances_onehotencoded

# Split dataset into train and dev sets
def splitDataset():
    X, y = prepareDataset()
    X_train, X_dev, y_train, y_dev = train_test_split(X,y, test_size = 0.2, random_state = 0)
    return X_train, X_dev, y_train, y_dev

def buildClassifier():
    classifier = Sequential()
    classifier.add(Dense(units = 128, kernel_initializer = "uniform", activation = "relu", input_dim = 300))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 128, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 4, kernel_initializer = "uniform", activation = "softmax"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

def runClassifier():
    X_train, X_dev, y_train, y_dev = splitDataset()
    classifier = KerasClassifier(build_fn = buildClassifier, batch_size = 32, epochs = 10)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
    mean = accuracies.mean()
    variance = accuracies.std()