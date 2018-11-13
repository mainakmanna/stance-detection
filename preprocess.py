import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def loadDataset(bodies_path, stances_path):
    """
    Given the file path of FNC bodies and stances path it loads the data.

    Args:
        bodies_path: path to bodies file.
        stances_path: path to stances file.

    Returns:
        headline_body_pairs: list of headline body pairs
        stances: list of stances        
    """
    headline_body_pairs = None
    stances = None
    if stances_path is not None:
        dataset_stances = pd.read_csv(stances_path)
        # array that holds individual stances for each headline-body pair
        stances = dataset_stances.iloc[:, 2]
        if bodies_path is not None:
            dataset_bodies = pd.read_csv(bodies_path)
            bodies_dict = dict(zip(dataset_bodies['Body ID'], dataset_bodies['articleBody']))
            headline_body_pairs = dataset_stances.iloc[:, 0:2].values
    
            for index in range(0, len(headline_body_pairs)):
                # replacing body id with actual body
                headline_body_pairs[index,1] = bodies_dict.get(headline_body_pairs[index,1])

    return headline_body_pairs, stances

def loadTestDataset():
    dataset_bodies = pd.read_csv("./dataset/competition_test_bodies.csv")
    dataset_stances = pd.read_csv("./dataset/competition_test_stances_unlabeled.csv")
    
    bodies_dict = dict(zip(dataset_bodies['Body ID'], dataset_bodies['articleBody']))
    headline_body_pairs = dataset_stances.iloc[:, 0:2].values
    
    for index in range(0, len(headline_body_pairs)):
        # replacing body id with actual body
        headline_body_pairs[index,1] = bodies_dict.get(headline_body_pairs[index,1])
    
    return headline_body_pairs
    
def loadDatasetGen():
    csv_file_path = "./dataset/train_stances.csv"
    c_size = 10000
    for dataset_stances in pd.read_csv(csv_file_path,chunksize=c_size):
        dataset_bodies = pd.read_csv("./dataset/train_bodies.csv")
        #dataset_stances = pd.read_csv("./dataset/train_stances.csv")
    
        # Forming dictionary of Body Id ---> Body.
        bodies_dict = dict(zip(dataset_bodies['Body ID'], dataset_bodies['articleBody']))
    
        # Takes the headline and body Id
        headline_body_pairs = dataset_stances.iloc[:, 0:2].values
        
        for index in range(0, len(headline_body_pairs)):
            # replacing body id with actual body
            headline_body_pairs[index,1] = bodies_dict.get(headline_body_pairs[index,1])
    
        # array that holds individual stances for each headline-body pair
        stances = dataset_stances.iloc[:, 2]
    
        yield headline_body_pairs, stances
        
def filterStopwords(tokenized_doc, filter = "all"):
    if filter == "all":
        stops = stopwords.words("english")
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?", "\n", "’", "``", "''", "...", "'", "\"", "'"])
    elif filter == "punctonly":
        stops = []
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "\n", "’", "``", "''", "...", "'"])
    
    stops = set(stops)
    return [w for w in tokenized_doc if (not w in stops)]

def tokenizeDoc(doc, stopwords = "all"):
    if stopwords == "none":
        return word_tokenize(doc.lower())
    return filterStopwords(word_tokenize(doc.lower()), stopwords)

def processDataset(stopwords = "all"):
    headline_body_pairs, stances = loadDataset()
    
    for i in range(0, len(headline_body_pairs)):
        headline_body_pairs[i][0] = tokenizeDoc(headline_body_pairs[i][0], stopwords)
        headline_body_pairs[i][1] = tokenizeDoc(headline_body_pairs[i][1], stopwords)
       
    return headline_body_pairs, stances

def writeProcessedDatasetToFileSystem():
    headline_body_pairs, stances = processDataset("all")
    
    with open("../dataset/headline_body_pairs.csv", 'w', encoding='utf-16', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ',')
        filewriter.writerow(["Headline", "Body"])
        for i in range(0, len(headline_body_pairs)):
            headline = " ".join(headline_body_pairs[i][0])
            body = " ".join(headline_body_pairs[i][1])
            filewriter.writerow([headline, body]) 
    
    with open("../dataset/stances.csv", 'w', encoding='utf-16', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ' ')
        filewriter.writerow(["Stance"])
        for i in range(0, len(stances)):
            stance = "".join(stances[i])
            filewriter.writerow([stance])
    