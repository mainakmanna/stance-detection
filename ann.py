import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    headline_body_pairs, stances = processDataset()
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
    X_train, X_dev, y_train, y_dev = train_test_split(X,y, test_size = 0.2, random_state = 0)
    print("Finished splitting dataset.")
    return X_train, X_dev, y_train, y_dev

def buildClassifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 128, kernel_initializer = "uniform", activation = "relu", input_dim = 600))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 128, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 4, kernel_initializer = "uniform", activation = "softmax"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

def runClassifier():
    X_train, X_dev, y_train, y_dev = splitDataset()
    print("Running classifier...")
    classifier = KerasClassifier(build_fn = buildClassifier("adam"), batch_size = 32, epochs = 10)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
    mean = accuracies.mean()
    variance = accuracies.std()
    return mean, variance
    
def tuneClassifier():
    X_train, X_dev, y_train, y_dev = splitDataset()
    print("Running classifier...")
    classifier = KerasClassifier(build_fn = buildClassifier)
    parameters = {'batch_size' : [25, 32, 64, 128], 
                  'epochs' : [10, 50, 100, 500],
                  'optimizer' : ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    return best_parameters, best_accuracy