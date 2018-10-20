def normalized_data_modeler(string):
    import csv
    import numpy as np
    from sklearn import preprocessing

    # Lists
    top = []
    bottom = []

    with open(string, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            top.append(row)

    # Transpose rows to columns
    top = np.array(top)
    top = top.T

    # Convert unique strings in each column to unique integers
    for i in top:
        middle = []
        unique = dict(enumerate(sorted(list(set(i)))))
        for j in i:
            middle.append((unique.keys()[unique.values().index(j)])*1.0)
        middle = preprocessing.scale(middle)
        bottom.append(middle)

    # Transpose columns back to rows
    bottom = np.array(bottom)
    samples = bottom[:len(bottom)-1].T
    visions = bottom[len(bottom)-1:].T
    data_model = zip(samples, visions)
    classes = len(set(sum(visions.tolist(), [])))
    return data_model, classes