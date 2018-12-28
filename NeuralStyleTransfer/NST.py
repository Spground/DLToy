#!/usr/bin/env python
# coding: utf-8
import os
import sys
import scipy.io
import scipy.misc
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.Session()

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [m, n_H*n_W, n_C]), perm=[0, 2, 1])# m * N_C * (n_H*n_W)
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [m, n_H*n_W, n_C]), perm=[0, 2, 1])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = 1/(4*n_H*n_W*n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]), perm=[1, 0]) # n_C * (n_H*n_W)
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]), perm=[1, 0])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = 1 / (4*n_C**2*(n_W*n_H)**2) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style
    return J

def model_nn(model, input_image, tensor_dic, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost 
        tensor_dic["train_step"].run(session=sess)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([tensor_dic["J"], tensor_dic["J_content"], tensor_dic["J_style"]])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


# Here's what the program will have to do:
# 1. Create an Interactive Session
# 2. Load the content image 
# 3. Load the style image
# 4. Randomly initialize the image to be generated 
# 5. Load the VGG16 model
# 7. Build the TensorFlow graph:
#     - Run the content image through the VGG16 model and compute the content cost
#     - Run the style image through the VGG16 model and compute the style cost
#     - Compute the total cost
#     - Define the optimizer and the learning rate
# 8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.
# 

def main(input_image_path, style_image_path, iteration=200):

    STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
    # content image

    content_image = scipy.misc.imread(input_image_path)
    content_img_shape = content_image.shape
    print(content_img_shape)
    content_image = reshape_and_normalize_image(content_image)
    
    style_image = scipy.misc.imread(style_image_path)
    style_image = scipy.misc.imresize(style_image, content_img_shape, interp='nearest')#keep same size as content image
    style_image = style_image[:,:,0:3] #3 channels
    style_image = reshape_and_normalize_image(style_image)
    print(style_image.shape)

    generated_image = generate_noise_image(content_image, shape=content_img_shape)#tensor
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat", shape=content_img_shape)
    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))
    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out
    
    # Compute the content cost
    tensor_dic = {}
    J_content = compute_content_cost(a_C, a_G)
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))
    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)
    J = total_cost(J_content, J_style)
    
    tensor_dic["J_content"] = J_content
    tensor_dic["J_style"] = J_style
    tensor_dic["J"] = J

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)
    tensor_dic["train_step"] = optimizer.minimize(J)

    model_nn(model, generated_image, tensor_dic, iteration)




