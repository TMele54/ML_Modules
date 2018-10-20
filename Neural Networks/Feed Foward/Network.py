def neural_network(data_model, classes, runs):
    # Python brain
    from pybrain.structure import FullConnection, FeedForwardNetwork, LinearLayer, SigmoidLayer, SoftmaxLayer
    from pybrain.datasets import ClassificationDataSet
    from pybrain.utilities import percentError
    from pybrain.supervised.trainers import BackpropTrainer
    from pybrain.tools.xml.networkwriter import NetworkWriter
    from pybrain.tools.xml.networkreader import NetworkReader
    import csv

    # Build Network
    try:

        n = NetworkReader.readFrom('resources/net.xml')
        print 'Loading previous network'

    except:

        print 'Generating new network'
        # Create a new Network
        n = FeedForwardNetwork()

        # Define the input layer
        inLayer = LinearLayer(len(data_model[0][0]))

        # Define a hidden layer
        hiddenLayer = SigmoidLayer(10)
        hiddenLayer2 = SigmoidLayer(10)

        # Define the output layer
        outLayer = LinearLayer(classes)

        # Add layers to network n
        n.addInputModule(inLayer)
        n.addModule(hiddenLayer)
        n.addModule(hiddenLayer2)
        n.addOutputModule(outLayer)

        # Create layers
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_hidden2 = FullConnection(hiddenLayer, hiddenLayer2)
        hidden2_to_out = FullConnection(hiddenLayer2, outLayer)

        # Add connectors to network n
        n.addConnection(in_to_hidden)
        n.addConnection(hidden_to_hidden2)
        n.addConnection(hidden2_to_out)

        # Finish Network
        n.sortModules()

    # Other Stuff

    ds = ClassificationDataSet(len(data_model[0][0]), 1, nb_classes=classes)
    # di = ClassificationDataSet(2,1,0)
    for o in data_model:
        ds.addSample(o[0], o[1])
    testing_data, training_data = ds.splitWithProportion(0.3)

    training_data._convertToOneOfMany()
    testing_data._convertToOneOfMany()

    print "Number of training patterns: ", len(training_data)
    print "Input and output dimensions: ", training_data.indim, training_data.outdim
    print "First sample (input, target, class):"
    print training_data['input'][0], training_data['target'][0], training_data['class'][0]

    trainer = BackpropTrainer(n, dataset=training_data)
    smart = []
    dumb = []

    with open("resources/minimum_error.csv", 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            smart.append(row)

    smart[0] = float(smart[0][0])
    print 'The minimum error from previous runs =', smart[0]

    for t in range(runs):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(), training_data['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=testing_data), testing_data['class'])
        print "epoch: %4d" % trainer.totalepochs, "  train error: %5.5f%%" % trnresult, " test error: %5.5f%%" % tstresult
        smart.append(tstresult)

        if tstresult <= min(smart):
            NetworkWriter.writeToFile(n, 'resources/net.xml')
            print 'Best!'
        else:
            dumb.append('1')
            print 'Worst!'

    minimum_error = []
    minimum_error.append(min(smart))

    with open("resources/minimum_error.csv", 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(minimum_error)

    print 'Minimum error (current state)', min(smart)
    return n