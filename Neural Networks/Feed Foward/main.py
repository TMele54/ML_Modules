from model import normalized_data_modeler
from Network import neural_network

# get data model
print 'Building Data Model...'
data_model, classes = normalized_data_modeler("resources/adult_data.csv")

# build network based on data model
print 'Learning...'

for i in range(100):
    print ''
    print ''
    print 'Run number:', i+1
    print ''
    net = neural_network(data_model, classes, 5)