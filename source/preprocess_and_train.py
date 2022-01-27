import json
from PIL.Image import SEQUENCE
import matplotlib
import matplotlib.pyplot as plt
from numpy.random.mtrand import shuffle
import cv2
import numpy as np
import scipy.ndimage as scpy
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import sys
import random
import os

# when printing numpy array then all will be printed
#np.set_printoptions(threshold=sys.maxsize)

# 32 x 32
IMG_HEIGHT = 32
IMG_WIDTH = 32

# Due to sensor placement it seems that first rows are always
# cold and unable to detect humans
DEL_ROW_AMNT = 8
# IMG_Y_RESIZED = int((IMG_HEIGHT - DEL_ROW_AMNT) * 0.75)
# IMG_X_RESIZED = int(IMG_WIDTH * 2.0 * 0.75)
IMG_Y_RESIZED = IMG_HEIGHT - DEL_ROW_AMNT
IMG_X_RESIZED = IMG_WIDTH

# Sensor 3078
S3078_FILE = '../dataset/thermal_raw_20210507_full/20210507_1605_3078.txt'
# Sensor C088
SC088_FILE = '../dataset/thermal_raw_20210507_full/20210507_1605_C088.txt'

s3078_data_arr = []
sc088_data_arr = []

human_images = []
background_images = []

x_train = []
y_train = []

x_test = []
y_test = []

def readSensorData():
    s3078_file = open(S3078_FILE, 'r')
    sc088_file = open(SC088_FILE, 'r')
    
    counter = 0
    while True:
        counter = counter + 1

        # Get one sample from the file
        s3078_sample = s3078_file.readline()
        sc088_sample = sc088_file.readline()

        # eof
        if (not s3078_sample or not sc088_sample):
            break

        if (counter % 4 == 0):
            # Convert sample into json form so it would be easier to parse
            s3078_json = json.loads(s3078_sample)
            sc088_json = json.loads(sc088_sample)

            # Get the data part from the sample
            s3078_data = np.array(s3078_json["data"])
            sc088_data = np.array(sc088_json["data"])

            s3078_data = np.delete(s3078_data, np.s_[0:DEL_ROW_AMNT], 0)
            sc088_data = np.delete(sc088_data, np.s_[0:DEL_ROW_AMNT], 0)

            s3078_data_arr.append(s3078_data)
            sc088_data_arr.append(sc088_data)

    # close sensor txt file
    s3078_file.close()
    sc088_file.close()

def removeHotPixels(img):
    image = np.copy(img)
    mean_temp = np.mean(image)
    for i, row in enumerate(image):
        for j, col in enumerate (row):
            if (image[i][j] > mean_temp):
                rand_float = (np.random.random() / 2) - 0.25
                image[i][j] = mean_temp - 0.5 + rand_float
    return image

def dataAugmentation():
    for sample in s3078_data_arr:
        # Human images
        human_images.append(sample)
        
        sample_cpy = np.copy(sample)
        sample_cpy = scpy.median_filter(sample_cpy, size=(3,3))
        human_images.append(sample_cpy)

        sample_cpy = np.copy(sample)
        sample_cpy = np.flip(sample_cpy, 1)
        human_images.append(sample_cpy)

        sample_cpy = scpy.median_filter(sample_cpy, size=(3,3))
        human_images.append(sample_cpy)

        # Background images
        sample_no_hot_pixels = removeHotPixels(sample)
        background_images.append(sample_no_hot_pixels)

        sample_no_hot_pixels_filtered = scpy.median_filter(sample_no_hot_pixels, size=(3,3))
        background_images.append(sample_no_hot_pixels_filtered)

        np.random.shuffle(sample_no_hot_pixels)
        background_images.append(sample_no_hot_pixels)

        sample_no_hot_pixels_filtered = scpy.median_filter(sample_no_hot_pixels, size=(3,3))
        background_images.append(sample_no_hot_pixels_filtered)

    for sample in sc088_data_arr:
        # Human images
        human_images.append(sample)
        
        sample_cpy = np.copy(sample)
        sample_cpy = scpy.median_filter(sample_cpy, size=(3,3))
        human_images.append(sample_cpy)

        sample_cpy = np.copy(sample)
        sample_cpy = np.flip(sample_cpy, 1)
        human_images.append(sample_cpy)

        sample_cpy = scpy.median_filter(sample_cpy, size=(3,3))
        human_images.append(sample_cpy)

        # Background images
        sample_no_hot_pixels = removeHotPixels(sample)
        background_images.append(sample_no_hot_pixels)

        sample_no_hot_pixels_filtered = scpy.median_filter(sample_no_hot_pixels, size=(3,3))
        background_images.append(sample_no_hot_pixels_filtered)

        np.random.shuffle(sample_no_hot_pixels)
        background_images.append(sample_no_hot_pixels)

        sample_no_hot_pixels_filtered = scpy.median_filter(sample_no_hot_pixels, size=(3,3))
        background_images.append(sample_no_hot_pixels_filtered)


def storeImages():
    for i, img in enumerate(human_images):
        # Multiplied by 10 in order not to lose precision
        # For example 13.4 will be 134 rather than 13
        img = img * 10
        cv2.imwrite("./imgs_human/img{}.png".format(i), img)

        # Resize images to be smaller
        #img = cv2.imread("imgs_human/img{}.png".format(i))
        #res = cv2.resize(img, (IMG_X_RESIZED, IMG_Y_RESIZED), interpolation = cv2.INTER_CUBIC)
        #cv2.imwrite("imgs_human_resized/img{}.png".format(i), img)

    for i, img in enumerate(background_images):
        # Multiplied by 10 in order not to lose precision
        # For example 13.4 will be 134 rather than 13
        img = img * 10 
        cv2.imwrite("./imgs_background/img{}.png".format(i), img)   
        
        # Resize images to be smaller
        #img = cv2.imread("imgs_background/img{}.png".format(i))
        #res = cv2.resize(img, (IMG_X_RESIZED, IMG_Y_RESIZED), interpolation = cv2.INTER_CUBIC)
        #cv2.imwrite("imgs_background_resized/img{}.png".format(i), img)

def prepareDataForTraining():
    global x_train
    global y_train
    global x_test
    global y_test

    training_data_prct = 0.8
    img_label_tuple = []
    
    for idx, im in enumerate(os.listdir("imgs_human/")):
        try:
            img_array = cv2.imread(os.path.join("imgs_human/", im))
            # Remove third dimension and divide by 10 to get original temp array
            img_array = np.array(img_array[:, :, 0]) / 10
            img_label_tuple.append((img_array, 1))
        except Exception as e:
            print("EXCEPTION")
            pass

    for idx, im in enumerate(os.listdir("imgs_background/")):
        try:
            img_array = cv2.imread(os.path.join("imgs_background/", im))
            # Remove third dimension and divide by 10 to get original temp array
            img_array = np.array(img_array[:, :, 0]) / 10
            img_label_tuple.append((img_array, 0))
        except Exception as e:
            print("EXCEPTION")
            pass

    random.shuffle(img_label_tuple)

    imgs, labels = zip(*img_label_tuple)
    training_amount = int((len(imgs) * training_data_prct))
    validation_amount = len(imgs) - training_amount

    x_train = np.array(imgs[:training_amount])
    y_train = np.array(labels[:training_amount])
    x_test = np.array(imgs[(-validation_amount):])
    y_test = np.array(labels[(-validation_amount):])

    # Normalize everything
    # x_train = tf.keras.utils.normalize(x_train)
    # x_test = tf.keras.utils.normalize(x_test)

    # TODO: something more reasonable perhaps
    x_train = x_train / 255
    x_test = x_test / 255
    
    x_train = np.array(x_train).reshape((-1, IMG_Y_RESIZED, IMG_X_RESIZED, 1))
    x_test = np.array(x_test).reshape((-1, IMG_Y_RESIZED, IMG_X_RESIZED, 1))


# TODO maybe: https://bleedai.com/human-activity-recognition-using-tensorflow-cnn-lstm/
def train():
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(IMG_Y_RESIZED, IMG_X_RESIZED, 1)))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    #model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(tf.keras.layers.Dense(2))

    model.summary()

    # Define parameters for training the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # BinaryCrossentropy
                  metrics=['accuracy'])
    
    # Train model - Adjust model parameters to minimize the loss and train it
    model.fit(x_train, y_train, epochs=2, batch_size=32)

    # Evaluate model performance
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print ("Validation evaluation results: loss - ", format(val_loss, '.3f'), "accuracy - ", format(val_acc, '.3f'))

    model.save('models/my_mnist.model')
    return model

def convertToTfLite(model):
    # https://www.tensorflow.org/lite/convert/index
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('models/model.tflite', 'wb') as f:
        f.write(tflite_model)

def runSomeInferenceTests(model):
    # TODO: run it on some unseen data
    predictions = model.predict(x_train[:10])
    print (y_train[:10])
    print (predictions)

def main():
    readSensorData()
    dataAugmentation()
    storeImages()
    prepareDataForTraining()
    model = train()
    convertToTfLite(model)
    runSomeInferenceTests(model)

if __name__ == "__main__":
    main()


# Write image to .txt file as C array

# with open('background.txt', 'w') as f:
#     counter = 0
#     for item in background_images:
#         for i in item:
#             f.write("{")
#             for j in i:
#                 f.write("%.4s, " % j)
#             f.write("},\n")
#         f.write("\n")
#         counter = counter +1
#         print (item)
#         if (counter >= 5):
#             break