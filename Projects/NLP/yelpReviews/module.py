import pandas as pd

files = [
    'yelp_academic_dataset_business.json',
    'yelp_academic_dataset_review.json'
]

pth = "/home/ubuntu/projects/yelp_data"

file_business = pth+files[0]
file_review = pth+files[1]

print 'Opening:', file_business, "&", file_review

data_businesses = pd.DataFrame(pd.read_json(file_business, lines=True))
data_reviews = pd.DataFrame(pd.read_json(file_review, lines=True))


print "first item"
print data_businesses.iloc[0]
print data_reviews.iloc[0]