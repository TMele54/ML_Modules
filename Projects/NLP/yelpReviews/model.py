########################################################################################################################
# Imports ##############################################################################################################
########################################################################################################################
from retrieveData import getCollectionData, writeCollectionData
from preProText import cleanTxt
from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import nltk, time, json
from pymongo import MongoClient
import pickle
from pprint import pprint

########################################################################################################################
# Word2Vec #############################################################################################################
########################################################################################################################
'''
query = {}
reviews = getCollectionData(query,'reviews')
sentences = []

print "Processing raw test & making tokenized collection"


def make_tokenizedReviewCollection(reviews):
    HOST, PORT = ('54.156.184.49', 27017)
    DATABASE_NAME = 'yelp_database'

    client = MongoClient(HOST, PORT)
    db = client["yelp_database"]
    col = db["reviews_tokenized"]
    review_number = 0
    temp=[]
    for review in reviews:
        res = col.find({"_id": review["_id"]}, {"_id": 1}).limit(1).count()
        if res != 1:
            reviewSentences = sent_tokenize(review["text"])

            for sentence in reviewSentences:
                sent = []
                reviewWords = word_tokenize(sentence)
                for word in reviewWords:
                    sent.append(word.lower())
                sentences.append(sent)

            review['text'] = sentences
            temp.append(review)
            if len(temp) >= 1000:
                print "Review Number:", review_number
                writeCollectionData(temp)
                temp = []
        else:
            print "Review,", review_number, " already exists in reviews_tokenized"
        review_number += 1

    return sentences


#continue to run to cleanse more reviews for analysis, movign foward now with subset ~14k
model_input_sentences = make_tokenizedReviewCollection(reviews)


tokenized_reviews = getCollectionData({},'reviews_tokenized')
max = 1000000 # One Million Sentences..
for review in tokenized_reviews:
    for sentence in review["text"]:
        sentences.append(sentence)

    if len(sentences) >= max:
        break

print len(sentences)

print "Sentences/Words Tokenized Complete"
print "pickling"
with open('review.pkl', 'wb') as f:
    pickle.dump(sentences, f)

'''

print "loading sentences from pickle"
with open('review.pkl', 'rb') as f:
    sentences = pickle.load(f)

print "Pre-processing text"
for i in range(0,len(sentences[0:9])):
    print sentences[i]

sentences = cleanTxt(sentences)

for i in range(0,len(sentences[0:9])):
    print sentences[i]


# Creating the model
print "Building Model, training on", len(sentences), "sentences"
model_word = Word2Vec(sentences, size=300, min_count=5, window=6, workers=8, sg=1, iter=10) #sg = 0 for CBOW, else skip gram
model_word.train(sentences, total_examples=len(sentences), epochs=100)
print "Model is generated & trained"

# Vocabulary
words = list(model_word.wv.vocab)

# Vector of love
tasty = model_word.wv["tasty"]
print("Tasty::", tasty)

# Saving the model
model_word.save("review.bin")

# Loading the saved model
model_word = Word2Vec.load("review.bin")

w1 = "man"
w2 = "women"
# Similarity between 2 words
print 'similar words man and women'
print(model_word.wv.similarity(w1, w2))


# Similar Words
print "words similar to pizza"
print(model_word.wv.most_similar(positive=["pizza"], topn=5))


# get the most common words
print "top 3 most common words"
print(model_word.wv.index2word[0], model_word.wv.index2word[1], model_word.wv.index2word[2])


# get the least common words
print "top 3 least common words"
vocab_size = len(model_word.wv.vocab)
print(model_word.wv.index2word[vocab_size - 1], model_word.wv.index2word[vocab_size - 2], model_word.wv.index2word[vocab_size - 3])


# find the index of the 2nd most common word ("of")
print "index of the word 'of'"
print('Index of "of" is: {}'.format(model_word.wv.vocab['of'].index))


# some similarity fun
print "similarity between words, man vs women - man vs elephant"
print(model_word.wv.similarity('woman', 'man'), model_word.wv.similarity('man', 'elephant'))


# what doesn't fit?
print "word that doesnt fit"
print(model_word.wv.doesnt_match("green red zebra".split()))

########################################################################################################################
# Doc2Vec ##############################################################################################################
########################################################################################################################
'''

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(doc,[self.labels_list[idx]])

sentencesLabels = []

it = LabeledLineSentence(sentences, sentencesLabels) # basically, just uses an iterable with the labels .. need to format 
# sentences for this..

model_doc = Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model_doc.build_vocab(it)

#training of model
for epoch in range(100):
    print 'iteration '+str(epoch+1)
    model_doc.train(it)
    model_doc.alpha -= 0.002
    model_doc.min_alpha = model_doc.alpha

#saving the created model
model_doc.save('doc2vec.model')
print "model saved"

#loading the model
d2v_model = Doc2Vec.load('doc2vec.model')
                                               
#start testing
#printing the vector of document at index 1 in docLabels
docvec = d2v_model.docvecs[1]
print docvec

#printing the vector of the file using its name
docvec = d2v_model.docvecs['1.txt'] #if string tag used in training
print docvec

#to get most similar document with similarity scores using document-index
similar_doc = d2v_model.docvecs.most_similar(14) 
print similar_doc

#to get most similar document with similarity scores using document- name
sims = d2v_model.docvecs.most_similar('1.txt')
print sims

#to get vector of document that are not present in corpus
docvec = d2v_model.docvecs.infer_vector('war.txt')
print docvec

'''