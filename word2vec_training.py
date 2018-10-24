import logging
from gensim.models import word2vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from preprocess import processDataset

# load Google's pre-trained word2vec model 
def loadWord2VecOnGoogleDataset():
    model = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin", binary = True)
    #result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    #print(result)
    return model

# converting Stanford's GloVe 840b 300d file format to word2vec file format
def convertGloveToWord2Vec():
    glove_input_file = "./models/glove.840b.300d.txt"
    word2vec_output_file = "./models/glove.840b.300d.word2vec.txt"
    glove2word2vec(glove_input_file, word2vec_output_file)
    
# load Stanford's 840b 300d pre-trained model      
def loadWord2VecConvertedFromGlove():
    model = KeyedVectors.load_word2vec_format("./models/glove.840b.300d.word2vec.txt")
    #result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    #print(result)
    return model

# train and save a word2vec model
def trainWord2VecModel(input, modelname):
    print("Starting word2vec training")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

    # set params
    num_features = 100      # Word vector dimensionality
    min_word_count = 5      # Minimum word count
    num_workers = 4         # Number of threads to run in parallel
    context = 5             # Context window size
    downsampling = 1e-3     # Downsample setting for frequent words
    trainalgo = 1           # cbow: 0 / skip-gram: 1

    print("Training model...")
    model = word2vec.Word2Vec(input, workers = num_workers, \
            size = num_features, min_count = min_word_count, \
            window = context, sample = downsampling, sg = trainalgo)

    # add for memory efficiency
    model.init_sims(replace=True)

    # save the model
    model.save(modelname)
    
# find most similar n words to given word
def applyWord2VecMostSimilar(modelname = "./models/GoogleNews-vectors-negative300.bin", word = "man", top = 10, pretrained = True):
    model = None
    if pretrained == True:
        model = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin", binary = True) 
    else:
        model = word2vec.Word2Vec.load(modelname)
        
    print("Find ", top, " terms most similar to ", word, "...")
    for res in model.most_similar(word, topn=top):
        print(res)
    print("Finding terms containing ", word, "...")
    for v in model.vocab:
        if word in v:
            print(v)

# train a word2vec model on the FNC dataset
def trainWord2VecOnFNCDataset():
    headline_body_pairs = processDataset("all")[0]
    inputModel = []
    for i in range(0, len(headline_body_pairs)):
        inputModel.append(headline_body_pairs[i][0])
        inputModel.append(headline_body_pairs[i][1])
    
    trainWord2VecModel(inputModel, "./models/testmodel")