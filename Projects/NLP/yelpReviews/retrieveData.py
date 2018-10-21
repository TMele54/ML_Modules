########################################################################################################################
# Imports ##############################################################################################################
########################################################################################################################

from pymongo import MongoClient
import json

########################################################################################################################
# MongoDB ##############################################################################################################
########################################################################################################################
HOST, PORT = ('54.156.184.49', 27017)
DATABASE_NAME = 'yelp_database'
COLLECTION_NAMES = ['businesses', 'reviews', 'checkings', 'tips', 'users']

client = MongoClient(HOST, PORT)
db = client["yelp_database"]

def getCollectionData(query):
    HOST, PORT = ('54.156.184.49', 27017)
    DATABASE_NAME = 'yelp_database'
    COLLECTION_NAMES = ['businesses', 'reviews', 'checkings', 'tips', 'users']

    client = MongoClient(HOST, PORT)
    db = client["yelp_database"]
    col = db[COLLECTION_NAMES[1]]
    print "Querying..."
    cursor = col.find(query)
    print "Query Resolved.."
    return cursor


def getBusinessReviews(db,cn):
    col = db[cn[0]]
    print "Querying.. ."
    cursor = col.aggregate([
                                {
                                    "$lookup": {
                                        "from": cn[1],
                                        "localField": "business_id",
                                        "foreignField": "business_id",
                                        "as": "reviews"
                                    }
                                }
                            ])
    print "Query Resolved.."
    return cursor


def writeCollectionData(data):
    HOST, PORT = ('54.156.184.49',27017)
    DATABASE_NAME = 'yelp_database'
    COLLECTION_NAMES = ['businesses', 'reviews', 'checkings', 'tips', 'users']

    client = MongoClient(HOST, PORT)
    db = client["yelp_database"]
    col = db[COLLECTION_NAMES[1] + "_tokenized"]

    print "Writing..."
    col.insert_one(data)
    print "Write Resolved.."

    return None

def dropCollection(col):
    HOST, PORT = ('54.156.184.49', 27017)
    DATABASE_NAME = 'yelp_database'
    COLLECTION_NAMES = ['businesses', 'reviews', 'checkings', 'tips', 'users']

    client = MongoClient(HOST, PORT)
    db = client["yelp_database"]
    db.drop_collection(col)

dropCollection("reviews_tokenized")


#businesses = getCollectionData(db,COLLECTION_NAMES[0],{})
#reviews = getCollectionData({})

#businessReviews = getBusinessReviews(db,[COLLECTION_NAMES[0],COLLECTION_NAMES[1]])
#for result in businessReviews:
#    print result
#    break