########################################################################################################################
# Imports ##############################################################################################################
########################################################################################################################

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

#for file in files:
#    for collection_name, path in file.items():
#        writeCollection(path, 'yelp_database', collection_name)
