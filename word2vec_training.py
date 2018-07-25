import logging
from gensim.models import word2vec, KeyedVectors
from preprocess import processDataset

def loadWord2VecOnGoogleDataset():
    model = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin", binary = True)
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    
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
def applyWord2VecMostSimilar(modelname = "./models/GoogleNews-vectors-negative300.bin", word, top=10, pretrained = True):
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
            
def main():
    headline_body_pairs = processDataset("all")[0]
    inputModel = []
    for i in range(0, len(headline_body_pairs)):
        inputModel.append(headline_body_pairs[i][0])
        inputModel.append(headline_body_pairs[i][1])
    
    trainWord2VecModel(inputModel, "./models/testmodel")
    applyWord2VecModel("./models/GoogleNews-vectors-negative300.bin", "man", 10, True)
            
if __name__ == '__main__':
    main()