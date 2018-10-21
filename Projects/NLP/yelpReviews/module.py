########################################################################################################################
# Imports ##############################################################################################################
########################################################################################################################

import pandas as pd
from pymongo import MongoClient
import json

data_directory = "/home/ubuntu/projects/yelp_data/"

file_names = [
    'yelp_academic_dataset_business.json',
    'yelp_academic_dataset_review.json',
    'yelp_academic_dataset_checkin.json',
    'yelp_academic_dataset_tip.json',
    'yelp_academic_dataset_user.json'
]

file_business = data_directory+file_names[0]
file_review = data_directory+file_names[1]
file_checkin = data_directory+file_names[2]
file_tip = data_directory+file_names[3]
file_user = data_directory+file_names[4]

files = [
    {"businesses": file_business},
    {"reviews": file_review},
    {"checkins": file_checkin},
    {"tips": file_tip},
    {"users": file_user}
]

########################################################################################################################
# MongoDB ##############################################################################################################
########################################################################################################################
'''
# Populate Database
def writeCollection(json_data, database, collection):

    HOST,PORT = ('54.156.184.49',27017)
    DATABASE_NAME = database
    COLLECTION_NAME = collection

    client = MongoClient(HOST,PORT)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    with open(json_data) as file:
        for line in file:
            print "***LINE:***", line
            collection.insert(json.loads(line))

for file in files:
    for collection_name, path in file.items():
        writeCollection(path, 'yelp_database', collection_name)
'''


'''
def joinCollections(query):

    HOST,PORT = ('54.156.184.49',27017)
    DATABASE_NAME = 'yelp_database'
    COLLECTION_NAMES = ['businesses','reviews']

    client = MongoClient(HOST,PORT)
    db = client[DATABASE_NAME]
    business_collection = db[COLLECTION_NAMES[0]]
    review_collection = db[COLLECTION_NAMES[1]]

    business_collection.aggregation([
        {
            "$lookup":{
                "from": review_collection,
                "localField" : "business_id",
                "foreignField" : "business_id",
                "as" : "[key name to appear in result]"
            }
        }
    ])
'''

########################################################################################################################
# Word2Vec #############################################################################################################
########################################################################################################################

# Reads alice.txt file
sample = open("C:\\Users\\Admin\\Desktop\\alice.txt", "r")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5)

# Print results
print("Cosine similarity between 'alice' and 'wonderland' - CBOW : ", model1.similarity('alice', 'wonderland'))


print("Cosine similarity between 'alice' and 'machines' - CBOW : ", model1.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5, sg=1)

# Print results
print("Cosine similarity between 'alice' and 'wonderland' - Skip Gram : ", model2.similarity('alice', 'wonderland'))


print("Cosine similarity between 'alice' and 'machines' - Skip Gram : ", model2.similarity('alice', 'machines'))
