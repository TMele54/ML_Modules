import os.path
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree

thisPath = os.path.abspath(os.path.dirname(__file__))
cars07 = "\\OpinRankDataset\\cars\\2007\\"
cars08 = "\\OpinRankDataset\\cars\\2008\\"
cars09 = "\\OpinRankDataset\\cars\\2009\\"

cars = [cars07, cars08, cars09]

for year in cars:
    data = thisPath+year
    files = [f for f in listdir(data) if isfile(join(data, f))]

    for file in files:
        print xml.etree.ElementTree.parse(file).getroot()
        item = {}
        item["type"] = file
        print item["type"]
        f = open(data+file, "rt")
        text = f.readlines()