import time
import csv
from itertools import zip_longest
import matplotlib as plt
import tensorflow as tf
import numpy as np
import os

RANDOM_SEED = 42
g = lambda x : 1/(1 + tf.exp(-x))

def nnScore(X, w, V, g):
    return tf.matmul(g((tf.matmul(X, w))), V)

def relu(x):
    y = x
    y[y < 0] = 0
    return y

def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):

    newfilePath = path+filename
    print ("Writing file to ", path+filename)
    poslist = positiveScores
    neglist = negativeScores

    # rows = zip(poslist, neglist)
    d = [poslist, neglist]
    export_data = zip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Normal", "Anomaly"))
        wr.writerows(export_data)
    myfile.close()

    return


def tf_OneClass_NN_linear(data_train,data_test,nu, verbose=True):

    tf.reset_default_graph()    
    tf.set_random_seed(RANDOM_SEED)

    train_X = data_train

    x_size = train_X.shape[1]  

    print  ("Input Shape:",x_size)

    h_size = 16   
    y_size = 1   
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    # nu = 0.1

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=1)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = (tf.matmul(X, w_1))  #
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : x

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu1(x):
        y = x
        y = tf.nn.relu(x)
        return y

    def relu(x):
        with sess.as_default():
            x = x.eval()
        y = x
        y[y < 0] = 0

        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)
        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g)))
        term4 = -r

        return term1 + term2 + term3 + term4

    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])
    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    cost = ocnn_obj(theta, X, nu, w_1, w_2, g,r)


    updates = tf.train.AdamOptimizer(0.05).minimize(cost)

    # Run optimization routine after initialization
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    start_time = time.time()

    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                rvalue = nnScore(train_X, w_1, w_2, g)
                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue, q=100*nu)
                if verbose:
                    print("Epoch = %d, r = %f" % (epoch + 1,rvalue))

 
    trainTime = time.time() - start_time
    ### Get the optimized weights here

    start_time = time.time()
    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    testTime = time.time() - start_time
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print ("====== Session Completed ======")


    pos_decisionScore = arrayTrain-rstar

    #pos_decisionScore[pos_decisionScore < 0] = 0 # why this?
    
    neg_decisionScore = arrayTest-rstar

    pos_decisionScore = pos_decisionScore.reshape(-1)
    neg_decisionScore = neg_decisionScore.reshape(-1) 

    write_decisionScores2Csv(os.getcwd()+'/Decision_Scores/', 'oc_nn_linear_cifar.csv', 
        pos_decisionScore, neg_decisionScore)

    # write_decisionScores2Csv(decision_scorePath, "OneClass_NN_linear.csv", pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]

def tf_OneClass_NN_sigmoid(data_train,data_test,nu, verbose=True):
    tf.reset_default_graph()
    sess = tf.Session()
    train_X = data_train
    tf.set_random_seed(RANDOM_SEED)

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    print  ("Input Shape:", x_size)
    h_size = 16                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    # nu = 0.1
        # def getActivations(layer, stimuli):
    #     units = sess.run(layer, feed_dict={x: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})
    #     plotNNFilter(units)

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : 1/(1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def data_rep(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        return g((tf.matmul(X, w)))

    def relu(x):
        y = tf.nn.relu(x)

        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)


        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4

    test_X = data_test
    X = tf.placeholder("float32", shape=[None, x_size])
    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # Run SGD

    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    start_time = time.time()
    for epoch in range(100):
            # Train with each example
                units = sess.run(updates, feed_dict={X: train_X,r:rvalue})
                # plotNNFilter(units)
                with sess.as_default():
                    w1 = w_1.eval()
                    w2 = w_2.eval()
                rvalue = nnScore(train_X, w1, w2, g)

                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*nu)
                if verbose:
                    print("Epoch = %d, r = %f" % (epoch + 1,rvalue))
    trainTime = time.time() - start_time
    with sess.as_default():
        w1 = w_1.eval()
        w2 = w_2.eval()



    start_time = time.time()
    train = nnScore(train_X, w1, w2, g)
    test = nnScore(test_X, w1, w2, g)
    train_rep = data_rep(train_X, w1, w2, g)
    test_rep = data_rep(test_X, w1, w2, g)

    testTime = time.time() - start_time
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
        arraytrain_rep =train_rep.eval()
        arraytest_rep= test_rep.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print ("====== Session Completed ======")

    pos_decisionScore = arrayTrain-rstar
    # pos_decisionScore[pos_decisionScore< 0] = 0 ## Clip all the negative values to zero
    neg_decisionScore = arrayTest-rstar

    pos_decisionScore = pos_decisionScore.reshape(-1)
    neg_decisionScore = neg_decisionScore.reshape(-1) 

    write_decisionScores2Csv(os.getcwd()+'/Decision_Scores/', 'oc_nn_sigmoid_cifar.csv', 
        pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]

def tf_OneClass_NN_relu(data_train,data_test,nu, verbose=True):
    tf.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(RANDOM_SEED)

    train_X = data_train
 
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    print  ("Input Shape:", x_size)
    h_size = 16                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    # nu = 0.1

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : relu(x)

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = tf.nn.relu(x)
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)

        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4

    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # Run SGD
    start_time = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                with sess.as_default():
                    w1 = w_1.eval()
                    w2 = w_2.eval()
                rvalue = nnScore(train_X, w1, w2, g)

                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*nu)
                if verbose:
                    print("Epoch = %d, r = %f" % (epoch + 1,rvalue))
    trainTime = time.time() - start_time
    with sess.as_default():
        w1 = w_1.eval()
        w2 = w_2.eval()

    start_time = time.time()
    train = nnScore(train_X, w1, w2, g)
    test = nnScore(test_X, w1, w2, g)
    testTime = time.time() - start_time

    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()

    rstar =rvalue
    sess.close()
    print ("====== Session Completed ======")

    pos_decisionScore = arrayTrain-rstar
    # pos_decisionScore[pos_decisionScore< 0] = 0 ## Clip all the negative values to zero
    neg_decisionScore = arrayTest-rstar

    pos_decisionScore = pos_decisionScore.reshape(-1)
    neg_decisionScore = neg_decisionScore.reshape(-1) 

    write_decisionScores2Csv(os.getcwd()+'/Decision_Scores/', 'oc_nn_sigmoid_relu.csv', 
        pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore, trainTime, testTime]

