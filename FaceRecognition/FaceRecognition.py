#!/usr/bin/env python
# coding: utf-8

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
import sys
from inception_blocks_v2 import *


np.set_printoptions(threshold=np.nan)

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))


    return loss

def load_database(FRmodel, db_img_dir=None):
    database = {}
    if not db_img_dir:
        database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
        database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
        database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
        database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
        database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
        database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
        database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
        database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
        database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
        database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
        database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
        database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
    else:
        img_files = [f for f in os.listdir(db_img_dir) if os.path.isfile(os.path.join(db_img_dir, f))
                and os.path.splitext(f)[1] and os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg", ".gif"]]
        print(img_files)
        for img in img_files:
            person_name = img.split(".")[0]
            print(person_name)
            print(os.path.join(db_img_dir, img))
            database[person_name] = img_to_encoding(os.path.join(db_img_dir, img), FRmodel)

    return database

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(database[identity] - encoding)

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 999999

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)
        print("name: %s, dis: %s" % (name, dist))
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

if __name__ == "__main__":
    input_image = sys.argv[1]
    db_img_dir = sys.argv[2]
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    database = load_database(FRmodel, db_img_dir)
    min_dist, identity = who_is_it(input_image, database, FRmodel)
    print(min_dist, identity)
