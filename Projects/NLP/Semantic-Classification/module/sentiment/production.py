########################################################################################################################
###################################################### Import ##########################################################
########################################################################################################################
### This script uses NLTK, stop-words, and SciKit Learn ###
import re, os, random
from collections import Counter
import pickle
import collections

# Stop words
from stop_words import get_stop_words

# Natural Language Tool Kit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.metrics import recall, precision, f_measure
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

# SkLearn
from sklearn import cross_validation

##################################################### Functions ########################################################
def get_reviews(directory): # walks directory of reviews, returns list of lists of review strings
    data = []
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            with open(directory+filename, 'r') as file:
                file_data = file.read().replace('\n', "")
                file.close()
                data.append(file_data)
    return data
def remove_special_char(reviews):# removes all punctuation excluding ' from reviews, return lol
    pattern = re.compile("[^\w']")
    for i in range(0,len(reviews)):
        reviews[i] = pattern.sub(' ', reviews[i])
    return reviews
def lower_case(reviews): # makes all letters lowercase, return lol
    for i in range(0,len(reviews)):
        reviews[i] = reviews[i].lower()
    return reviews
def tokenize(reviews): # parses text on whitespace to ionize words, returns a list of lists that contains tokens
    for i in range(0,len(reviews)):
        reviews[i] = reviews[i].split()
    return reviews
def remove_stop_words(tokens): # removes words that are regarded not specific to any particular meaning [unigram]
    stop_words = get_stop_words('en')
    stop_words = set(stopwords.words('english')+stop_words)
    good_token_list = []
    for token_set in tokens:
        good_tokens = []
        for token in token_set:
            if token in stop_words:
                pass
            else:
                good_tokens.append(token)
        good_token_list.append(good_tokens)

    return good_token_list
def stemming(tokens): # remove all word extensions such that running, run and runner all return 'run'
    porter_stemmer = PorterStemmer()
    for i in range(0, len(tokens)):
        for j in range(0, len(tokens[i])):
            tokens[i][j] = porter_stemmer.stem(tokens[i][j])

    return tokens
def bag_of_words(tokens): # aggregates all words into a single list
    bag = []
    for token_set in tokens:
        for token in token_set:
            bag.append(token)
    return bag
def top_words(bag, top): # nests the bog of words into a counter set of word frequencies and thresholds the length of the array by frequency
    counted = Counter(bag)
    top_talkers = counted.most_common(top)
    return top_talkers
def remove_numbers(tokens):  # removes numbers
    for i in range(0, len(tokens)):
        tokens[i] = [word for word in tokens[i] if not word.isdigit()]
    return tokens
def word_feats(words): # transforms the list of formatted token lists into dictionaries of the same with key value    'word': True
    return dict([(word, True) for word in words])
def save_classifier(classifier): # saves the classifier in a binary
    f = open('classifier/my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
def load_classifier():  # loads the classifuer from binary
    f = open('classifier/my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
def features(processed_text, descriptor): # utilizes word_feats to iterate over all reviews and allows for custom formmatting
    return [(word_feats(review), descriptor) for review in processed_text]
def test_train(set_a, set_b): # splits the formatted dictionaries into 2 sets, one for training, and a second for testing based on a fraction
    break_point_a = len(set_a) * 3 / 4
    break_point_b = len(set_b) * 3 / 4
    train = set_b[:break_point_b] + set_a[:break_point_a]
    test = set_b[break_point_b:] + set_a[break_point_a:]
    return test, train
def make_ngrams(input_list, n):
    if n > 0:
        if n == 1:
            return input_list
        else:
            for j in range(len(input_list)):
                input_list[j] = zip(*[input_list[j][i:] for i in range(n)])
            return input_list
    else:
        print "ngrams must be greater than 0"

########################################################################################################################
################################################### Get Raw Data #######################################################
########################################################################################################################
review_number = 280
n = 2 # 1 for unigram, 2 for bigram, 3 for trigram, and so on
base = "../../data/"
_set = ["emails/", "movie_reviews/"]
ext = ["ham/", "spam/", "pos/", "neg/"]

typeA = "positive"
typeB = "negative"

pos_reviews = get_reviews(base + _set[1] + ext[2])
neg_reviews = get_reviews(base + _set[1] + ext[3])
#neg_reviews = neg_reviews[:1500]

########################################################################################################################
############################################### Deploy Classifier ######################################################
########################################################################################################################
print "remove and load old classifier"
classifier = None

classifier = load_classifier()

print "classify text"
string = "not amazing"
text = [string]
f_text = remove_special_char(text)
f_text = lower_case(f_text)
f_text = tokenize(f_text)
f_text = make_ngrams(f_text, n)

print f_text

classify_me = [word_feats(review) for review in f_text]
print
print "For the string:"
print
print string
print
print "The Bayes Classifer labels our string as '", classifier.classify(classify_me[0]),"'"
