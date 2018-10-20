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
n = 1 # 1 for unigram, 2 for bigram, 3 for trigram, and so on
base = "../../data/"
_set = ["emails/", "movie_reviews/"]
ext = ["ham/", "spam/", "pos/", "neg/"]

typeA = "positive"
typeB = "negative"

pos_reviews = get_reviews(base + _set[1] + ext[2])
neg_reviews = get_reviews(base + _set[1] + ext[3])
#neg_reviews = neg_reviews[:1500]

print
''' Example output '''
print "A review looks like this:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print


########################################################################################################################
################################################# Pre-processing #######################################################
########################################################################################################################

### Special Characters ###
pos_reviews = remove_special_char(pos_reviews)
neg_reviews = remove_special_char(neg_reviews)

''' Example output '''
print "Review after removing special character:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print

### Lower Casing ###
pos_reviews = lower_case(pos_reviews)
neg_reviews = lower_case(neg_reviews)


''' Example output '''
print "Lowercase review:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print

### Tokenization ###
pos_reviews = tokenize(pos_reviews)
neg_reviews = tokenize(neg_reviews)

''' Example output '''
print "Tokenized review:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print


### N-gramming ###
pos_reviews = make_ngrams(pos_reviews,n)
neg_reviews = make_ngrams(neg_reviews,n)

''' Example output '''
print "nGram review:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print


### Remove Numbers ###
# pos_reviews = remove_numbers(pos_reviews)
# neg_reviews = remove_numbers(neg_reviews)
''' Example output
print "Numbers removed:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print
'''

### Randomization ###
# pos_reviews = random.sample(pos_reviews, len(pos_reviews))
# neg_reviews = random.sample(neg_reviews, len(neg_reviews))
''' Example output
print "Randomized order of reviews:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print
'''

### Remove stops ###
# pos_reviews = remove_stop_words(pos_reviews)
# neg_reviews = remove_stop_words(neg_reviews)
''' Example output
print "Stop words removed:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print
'''

### Stemming ###
# pos_reviews = stemming(pos_reviews)
# neg_reviews = stemming(neg_reviews)
''' Example output
print "stemmed review:"
print "############################################################"
print pos_reviews[review_number]
print "############################################################"
#raw_input('Press Enter to continue')
print
'''

### Bag of words ###
# pos_bag = bag_of_words(pos_reviews)
# neg_bag = bag_of_words(neg_reviews)
''' Example output
print "Bag of words:"
print "############################################################"
print pos_bag
print "############################################################"
#raw_input('Press Enter to continue')
print
'''

### Frequent words ###
# top = 10
# pos_top = top_words(pos_bag, top)
# neg_top = top_words(neg_bag, top)
''' Example output
print "Tokenized review:"
print "############################################################"
print pos_top
print "############################################################"
#raw_input('Press Enter to continue')
print
'''

########################################################################################################################
############################################## Exploratory Analysis ####################################################
########################################################################################################################

'''Done by visual inspection'''

########################################################################################################################
################################################ Bayes Classifier ######################################################
########################################################################################################################
positive_features = features(pos_reviews, typeA)
negative_features = features(neg_reviews, typeB)

test, train = test_train(positive_features, negative_features)

print 'train on %d instances, test on %d instances' % (len(train), len(test))

classifier = NaiveBayesClassifier.train(train)

reference_sets = collections.defaultdict(set)
test_sets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test):
    reference_sets[label].add(i)
    observed = classifier.classify(feats)
    test_sets[observed].add(i)

print 'Accuracy:', nltk.classify.util.accuracy(classifier, test)
print 'Positive precision:', precision(reference_sets[typeA], test_sets[typeA])
print 'Positive recall:',    recall(reference_sets[typeA],    test_sets[typeA])
print 'Positive F-measure:', f_measure(reference_sets[typeA], test_sets[typeA])
print 'Negative precision:', precision(reference_sets[typeB], test_sets[typeB])
print 'Negative recall:',    recall(reference_sets[typeB],    test_sets[typeB])
print 'Negative F-measure:', f_measure(reference_sets[typeB], test_sets[typeB])
print ""
classifier.show_most_informative_features()


print "Save classifier"
save_classifier(classifier)

########################################################################################################################
################################################ Cross Validation ######################################################
########################################################################################################################

print 'Performing cross validation:'
folds = 3
cv = cross_validation.KFold(len(train), n_folds=folds, shuffle=False, random_state=None)

accuracies = []
pos_pres = []
pos_recs = []
pos_fs = []
neg_pres = []
neg_recs = []
neg_fs = []
counter = 1
print cv
for cross_validation_training, cross_validation_testing in cv:
    print "Performing fold number", counter
    counter += 1
    classifier = nltk.NaiveBayesClassifier.train(train[cross_validation_training[0]:cross_validation_training[len(cross_validation_training)-1]])
    acc = nltk.classify.util.accuracy(classifier, train[cross_validation_testing[0]:cross_validation_testing[len(cross_validation_testing)-1]])

    reference_sets = collections.defaultdict(set)
    test_sets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test):
        reference_sets[label].add(i)
        observed = classifier.classify(feats)
        test_sets[observed].add(i)

    pos_pre = precision(reference_sets[typeA], test_sets[typeA])
    pos_rec = recall(reference_sets[typeA], test_sets[typeA])
    pos_f = f_measure(reference_sets[typeA], test_sets[typeA])
    neg_pre = precision(reference_sets[typeB], test_sets[typeB])
    neg_rec = recall(reference_sets[typeB], test_sets[typeB])
    neg_f = f_measure(reference_sets[typeB], test_sets[typeB])

    accuracies.append(acc)
    pos_pres.append(pos_pre)
    pos_recs.append(pos_rec)
    pos_fs.append(pos_f)
    neg_pres.append(neg_pre)
    neg_recs.append(neg_rec)
    neg_fs.append(neg_f)

    print 'Positive precision:', pos_pre
    print 'Positive recall:', pos_rec
    print 'Positive F-measure:', pos_f
    print 'Negative precision:', neg_pre
    print 'Negative recall:', neg_rec
    print 'Negative F-measure:', neg_f
    print 'accuracy:', acc
    print

print
print "The average accuracy is", sum(accuracies)/folds
print 'The average Positive precision:', sum(pos_pres)/folds
print 'The average Positive recall:', sum(pos_recs)/folds
print 'The average Positive F-measure:', sum(pos_fs)/folds
print 'The average Negative precision:', sum(neg_pres)/folds
print 'The average Negative recall:', sum(neg_recs)/folds
print 'The average Negative F-measure:', sum(neg_fs)/folds
print

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

classify_me = [word_feats(review) for review in f_text]
print
print "For the string:"
print
print string
print
print "The Bayes Classifer labels our string as '", classifier.classify(classify_me[0]),"'"
