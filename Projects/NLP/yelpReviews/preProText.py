########################################################################################################################
# Clean Text ###########################################################################################################
########################################################################################################################
# remove digits, stops, 1 letter words, special characters, names etc
def cleanTxt(list_of_sentences):
    from nltk.corpus import stopwords
    import re

    # Stop Words
    stop_words = stopwords.words('english')
    for i in range(0,len(stop_words)):
        stop_words[i] = stop_words[i].encode('utf-8')

    stops = []

    with open("additional_stops.txt", 'r') as f:
        for word in f.readlines():
            stops.append(word.strip())

    _stop_words = set(stop_words+stops)

    # Regex for Special Characters
    pattern = re.compile("[^\w']")
    perm = []
    temp = []
    for i in range(0,len(list_of_sentences)):
        temp = []
        for j in range(0,len(list_of_sentences[i])):
            # Lowercase
            list_of_sentences[i][j] = list_of_sentences[i][j].lower()

            # Remove Stops
            if list_of_sentences[i][j] in stop_words:
                pass
            elif len(list_of_sentences[i][j]) == 1:
                pass
            elif list_of_sentences[i][j].isdigit() == True:
                pass
            else:
                temp.append(list_of_sentences[i][j])

        perm.append(temp)

    return perm