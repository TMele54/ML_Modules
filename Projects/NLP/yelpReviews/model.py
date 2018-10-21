########################################################################################################################
# Imports ##############################################################################################################
########################################################################################################################
from retrieveData import getCollectionData, writeCollectionData
from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import nltk, time, json
from pymongo import MongoClient

########################################################################################################################
# Word2Vec #############################################################################################################
########################################################################################################################
start = time.time()
query = {}
reviews = getCollectionData(query)
sentences = []

print "Processing raw test & making tokenized collection"

now = time.time()
print("Time::",now - start)

def make_tokenizedReviewCollection():
    HOST, PORT = ('54.156.184.49', 27017)
    DATABASE_NAME = 'yelp_database'

    client = MongoClient(HOST, PORT)
    db = client["yelp_database"]
    col = db["reviews_tokenized"]

    for review in reviews:
        res = col.find({"_id": review["_id"]}, {"_id": 1}).limit(1).count()
        if res[0] == 1:
            pass
        else:
            reviewSentences = sent_tokenize(review["text"])

            for sentence in reviewSentences:
                sent = []
                reviewWords = word_tokenize(sentence)
                for word in reviewWords:
                    sent.append(word.lower())
                sentences.append(sent)

            review['text'] = sentences  
            writeCollectionData(review)
            
    return sentences

model_input_sentences = make_tokenizedReviewCollection()

print "Sentences/Words Tokenized Complete"
now = time.time()
print(now - start)

# Creating the model
print "Building Model"
model = Word2Vec(model_input_sentences, size=150, min_count=4, window=5, workers=4, sg=0) #sg = 0 for CBOW, else skip gram
print "Model is generated"
now = time.time()
print(now - start)

# Saving the model
print "Saving Model"
now = time.time()
print(now - start)
model.save("reviews.bin")

# Loading the saved model
print "Loading Model"
model = Word2Vec.load("reviews.bin")
now = time.time()
print(now - start)
# Vocabulary
words = list(model.wv.vocab)

# Vector of Alice
alice = model.wv["Alice"]
print("Alice::", alice)

# Saving the model
model.save("alice.bin")

# Loading the saved model
model = Word2Vec.load("alice.bin")

# Similarity between 2 words
print(model.wv.similarity("Alice", "Rabbit"))
print(model.wv.similarity("That", "baby"))

# Similar Words
print(model.wv.most_similar(positive=["Alice", "cat"], negative=["baby"]))


'''

import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from tqdm import tqdm
from copy import deepcopy
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from string import punctuation
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# importing bokeh library for interactive dataviz
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from sklearn.preprocessing import scale

LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

tqdm.pandas(desc="progress-bar")

tokenizer = TweetTokenizer()

def ingest():
    data = pd.read_csv('./tweets.csv')
    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map(int)
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data

data = ingest()
data.head(5)

def tokenize(tweet):
    try:
        tweet = unicode(tweet.decode('utf-8').lower())
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'


def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

x_train, x_test, y_train, y_test = train_test_split(
    np.array(data.head(n).tokens),np.array(data.head(n).Sentiment), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

x_train[0]

TaggedDocument(words=[u'thank', u'you', u'!', u'im', u'just', u'a', u'tad', u'sad', u'u', u'r', u'off', u'the', u'market', u'tho', u'...'], tags=['TRAIN_0'])

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)])

tweet_w2v['good']

tweet_w2v.most_similar('good')


# defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in tweet_w2v.wv.vocab.keys()[:5000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = tweet_w2v.wv.vocab.keys()[:5000]

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf)

print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print score[1]
'''