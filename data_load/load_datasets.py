from sklearn import utils
import dataset
import numpy as np
import cv2
from imutils import paths
import os
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,Activation
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input
 
nu = 0.001
activations = ["linear", "rbf"]
dataPath = "./data/"

def readjpegimages2Array(filepath):
    from PIL import Image
    
    folder = filepath
    read = lambda imname: np.asarray(Image.open(imname))
    ims = [np.array(Image.open(os.path.join(folder, filename))) for filename in os.listdir(folder)]
    imageList = []
    for x in range(0,len(ims)):

        if(ims[x].shape ==(256,256)):
            imageList.append(ims[x])
    result = np.asarray(imageList)

    return result

def prepare_cifar_10_data():

    from tflearn.datasets import cifar10

    image_and_anamolies = {'image': 5, 'anomalies1': 3, 'anomalies2': 3, 'imagecount': 220, 'anomaliesCount': 11}

    def prepare_cifar_data_with_anamolies(original, original_labels, image_and_anamolies):

        imagelabel = image_and_anamolies['image']
        imagecnt = image_and_anamolies['imagecount']

        idx = np.where(original_labels == imagelabel)

        idx = idx[0][:imagecnt]

        data = original[idx]
        data_labels = original_labels[idx]

        anamoliescnt = image_and_anamolies['anomaliesCount']
        anamolieslabel1 = image_and_anamolies['anomalies1']

        anmolies_idx1 = np.where(original_labels == anamolieslabel1)
        anmolies_idx1 = anmolies_idx1[0][:(anamoliescnt)]

        ana_images1 = original[anmolies_idx1]
        ana_images1_labels = original_labels[anmolies_idx1]

        anamolieslabel2 = image_and_anamolies['anomalies2']
        anmolies_idx2 = np.where(original_labels == anamolieslabel2)
        anmolies_idx2 = anmolies_idx2[0][:(anamoliescnt)]

        anomalies = original[anmolies_idx2]
        anomalies_labels = original_labels[anmolies_idx2]

        return [data, data_labels, anomalies, anomalies_labels]

    # load cifar-10 data
    ROOT = dataPath + 'cifar-10_data/'
    (X, Y), (testX, testY) = cifar10.load_data(ROOT)
    testX = np.asarray(testX)
    testY = np.asarray(testY)

    [train_X, train_Y, testX, testY] = prepare_cifar_data_with_anamolies(X, Y, image_and_anamolies)

    # Make the Dog samples as positive and cat samples as negative
    testY[testY == 5] = 1
    testY[testY == 3] = -1
    test_y = [[i] for i in testY]

    # Modify to suite the tensorflow code
    train_Y[train_Y == 5] = 1
    train_Y[train_Y == 3] = -1
    train_y = [[i] for i in train_Y]

    train_X = np.reshape(train_X, (len(train_X), 3072))
    testX = np.reshape(testX, (len(testX), 3072))

    # print "Data Train Shape",train_X.shape
    # print "Data Test Shape",testX.shape

    # print "Label Train Shape",train_Y.shape
    # print "Label Test Shape",testY.shape

    data_train = train_X
    data_test = testX
    targets_train = train_Y
    targets_test = testY

    train_X = train_X
    test_X = testX
    train_y = train_Y
    test_y = testY
    train_y = train_y.tolist()
    train_y = [[i] for i in train_y]

    test_y = test_y.tolist()
    test_y = [[i] for i in test_y]

    return [train_X, train_y, test_X, test_y]


def prepare_cifar_10_data_for_conv_net():

    # Prepare input data
    classes = ['dogs', 'cats']
    num_classes = len(classes)
    # 20% of the data will automatically be used for validation
    validation_size = 0.0
    img_size = 128
    num_channels = 3
    train_path = dataPath + '/cifar-10_data/train/'
    test_path = dataPath + '/cifar-10_data/test/'

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    data = dataset.read_train_sets(train_path, img_size, classes, validation_size=0)
    test_images, test_ids = dataset.read_test_set(test_path, img_size)
    # print("Size of:")
    # print("- Training-set:\t\t{}".format(len(data.train.images)))
    # print("- Test-set:\t\t{}".format(len(test_images)))
    # print("- Validation-set:\t{}".format(len(data.valid.labels)))

    return [data.train.images, data.train.labels, test_images, test_ids, data]

def add_new_last_layer(base_model_output,base_model_input):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model_output
  print ("base_model.output",x.shape)
  inp = base_model_input
  print ("base_model.input",inp.shape)
  dense1 = Dense(512, name="dense_output1")(x)  # new sigmoid layer
  dense1out = Activation("relu", name="output_activation1")(dense1)
  dense2 = Dense(1, name="dense_output2")(dense1out) #new sigmoid layer
  dense2out = Activation("relu",name="output_activation2")(dense2)  # new sigmoid layer
  model = Model(inputs=inp, outputs=dense2out)
  return model


def prepare_cifar_data_for_rpca_forest_ocsvm(train_path,test_path):

    print("[INFO] loading images for training...")
    data = []
    data_test = []
    labels = []
    labels_test = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(train_path)))
    # loop over the input training images
    image_dict = {}
    test_image_dict = {}
    i = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (1, 3072))
        image = img_to_array(image)
        data.append(image)
        image_dict.update({i: imagePath})
        i = i + 1

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "dogs" else 0
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data)

    y = keras.utils.to_categorical(labels, num_classes=2)
    # print image_dict

    print("[INFO] preparing test data (anomalous )...")
    testimagePaths = sorted(list(paths.list_images(test_path)))
    # loop over the test images
    j = 0
    for imagePath in testimagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (1, 3072))
        image = img_to_array(image)
        data_test.append(image)
        test_image_dict.update({j: imagePath})
        j = j + 1
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 0 if label == "cats" else 1
        labels_test.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data_test = np.array(data_test)

    ## Normalise the data
    data = np.array(data) / 255.0
    data = np.reshape(data,(len(data),3072))
    data_test = np.array(data_test) / 255.0
    data_test = np.reshape(data_test, (len(data_test), 3072))




    return [data,data_test]

def prepare_cifar_data_for_cae_ocsvm(train_path,test_path,modelpath):
    ### Declare the training and test paths
    # modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
    #
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"
    NB_IV3_LAYERS_TO_FREEZE = 23
    side = 32
    side = 32
    channel = 3
    keras.backend.clear_session()
    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print ("base_model.output", x.shape)
        inp = base_model_input
        print ("base_model.input", inp.shape)
        dense1 = Dense(512, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print ("Length of model layers...", len(model.layers))

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    print("[INFO] loading images for training...")
    data = []
    data_test = []
    labels = []
    labels_test = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(train_path)))
    # loop over the input training images
    image_dict = {}
    test_image_dict = {}
    i = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data.append(image)
        image_dict.update({i: imagePath})
        i = i + 1

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "dogs" else 0
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data)

    y = keras.utils.to_categorical(labels, num_classes=2)
    # print image_dict

    print("[INFO] preparing test data (anomalous )...")
    testimagePaths = sorted(list(paths.list_images(test_path)))
    # loop over the test images
    j = 0
    for imagePath in testimagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data_test.append(image)
        test_image_dict.update({j: imagePath})
        j = j + 1
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 0 if label == "cats" else 1
        labels_test.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data_test = np.array(data_test)

    ## Normalise the data
    data = np.array(data) / 255.0
    data_test = np.array(data_test) / 255.0



    # Preprocess the inputs
    x_train = data
    x_test = data_test

    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(x_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(x_test)
    X_test = intermediate_output   

    K.clear_session()

    return [X,X_test]

def prepare_cifar_data_for_ae2_svdd(train_path,test_path,modelpath):
    
    NB_IV3_LAYERS_TO_FREEZE = 4
    side = 32
    side = 32
    channel = 3

    ## Declare the scoring functions
    g = lambda x: 1 / (1 + tf.exp(-x))

    # g  = lambda x : x # Linear
    def nnScore(X, w, V, g):

        # print "X",X.shape
        # print "w",w[0].shape
        # print "v",V[0].shape
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        y[y < 0] = 0
        return y

    def add_new_last_layer(base_model_output, base_model_input):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model_output
        print ("base_model.output", x.shape)
        inp = base_model_input
        print ("base_model.input", inp.shape)
        dense1 = Dense(128, name="dense_output1")(x)  # new sigmoid layer
        dense1out = Activation("relu", name="output_activation1")(dense1)
        dense2 = Dense(1, name="dense_output2")(dense1out)  # new sigmoid layer
        dense2out = Activation("relu", name="output_activation2")(dense2)  # new sigmoid layer
        model = Model(inputs=inp, outputs=dense2out)
        return model

    # load the trained convolutional neural network freeze all the weights except for last four layers
    print("[INFO] loading network...")
    model = load_model(modelpath)
    model = add_new_last_layer(model.output, model.input)
    print(model.summary())
    print ("Length of model layers...", len(model.layers))

    # Freeze the weights of untill the last four layers
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    print("[INFO] loading images for training...")
    data = []
    data_test = []
    labels = []
    labels_test = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(train_path)))
    # loop over the input training images
    image_dict = {}
    test_image_dict = {}
    i = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data.append(image)
        image_dict.update({i: imagePath})
        i = i + 1

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "dogs" else 0
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data)

    y = keras.utils.to_categorical(labels, num_classes=2)
    # print image_dict

    print("[INFO] preparing test data (anomalous )...")
    testimagePaths = sorted(list(paths.list_images(test_path)))
    # loop over the test images
    j = 0
    for imagePath in testimagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (side, side))
        image = img_to_array(image)
        data_test.append(image)
        test_image_dict.update({j: imagePath})
        j = j + 1
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 0 if label == "cats" else 1
        labels_test.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    data_test = np.array(data_test)

    ## Normalise the data
    data = np.array(data) / 255.0
    data_test = np.array(data_test) / 255.0



    # Preprocess the inputs
    x_train = data
    x_test = data_test

    ## Obtain the intermediate output
    layer_name = 'dense_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    layer1 = model.get_layer("dense_output1")
    layer2 = model.get_layer("dense_output2")
    intermediate_output = intermediate_layer_model.predict(x_train)
    X = intermediate_output
    intermediate_output = intermediate_layer_model.predict(x_test)
    X_test = intermediate_output
 

    K.clear_session()

    return [X,X_test]




