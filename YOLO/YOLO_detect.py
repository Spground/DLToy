import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
#import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import subprocess
import shutil

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs  #19 * 19 * 5 * 80
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis = -1)  #19 * 19 * 5 * 1
    box_class_scores = K.max(box_scores, axis = -1) #19 * 19 * 5 * 1
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >=  threshold  #19 * 19 * 5 * 1
    
    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask) # ? * 1
    boxes = tf.boolean_mask(boxes, filtering_mask)  # ?  * 4
    classes = tf.boolean_mask(box_classes, filtering_mask) # ? * 1
    
    return scores, boxes, classes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    nms_indices = tf.image.non_max_suppression(boxes = boxes, scores = scores, max_output_size = max_boxes_tensor,
                                               iou_threshold = iou_threshold)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, 
    box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) 
    (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs # 19 * 19 * 5 * 1,19 * 19 * 5 * 2, 19 * 19 * 5 * 2,19 * 19 * 5 * 80

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh) # 19 * 19 * 5 * 4 (x_mi, y_mid,w,h) convets to (x1,y1,x2,y2)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)# None * 1, None * 4, None * 1
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape) # None * 4

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold) # None * 1, None * 4. None * 1
    
    return scores, boxes, classes

def predict(sess, image_file, tensor = None):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run(tensor, 
                                                  feed_dict = {yolo_model.input: image_data, K.learning_phase():0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", os.path.basename(image_file)), quality=90)
    # Display the results in the notebook
    #output_image = scipy.misc.imread(os.path.join("out", image_file))
    #imshow(output_image)
    
    return out_scores, out_boxes, out_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO detect')
    parser.add_argument('--file', '-f', help='input video file path')
    parser.add_argument('--width', '-w', type=int, help="video width")
    parser.add_argument('--height', '-H', type=int, help="video height")
    parser.add_argument('--rate', '-r', default=29.97, help="sample rate")
    parser.add_argument('--out_dir', '-o', help='output dir')
    
    args = parser.parse_args()
    print(args)
    if len(sys.argv) <= 1:
        parser.print_help()
        exit(1)
    
    #args
    video_file= args.file #video path
    w = args.width #width
    h = args.height #height
    
    rate = args.rate #sample rate
    out_dir = args.out_dir# video output dir

    if not os.path.exists(video_file):
        print("video:%s does not exits." % video_file)
        exit(1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #create temp dir
    tmp_image_dir = "tmp"
    if os.path.exists(tmp_image_dir):
        shutil.rmtree(tmp_image_dir)
    os.makedirs(tmp_image_dir)
    cmd = ['./scripts/split_video.sh', video_file, rate, tmp_image_dir]
    print(cmd)
    code = subprocess.call(cmd)
    if code == 0:
        print("video spliting done.")
    else:
        print("something wrong, code is %d" % code)
        exit(1)
    #check image length
    image_len = len([f for f in os.listdir(tmp_image_dir) if os.path.isfile(os.path.join(tmp_image_dir, f))
                     and os.path.splitext(f)[1] == ".jpg"])
    print("image lenght is %d " % image_len)
    
    sess = K.get_session()
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    image_shape = (float(h), float(w))
    
    yolo_model = load_model("model_data/yolo.h5")
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    if not os.listdir("out"):
        for i in range(1, image_len + 1):
            predict(sess, os.path.join(tmp_image_dir, "image-%08d.jpg" % i), (scores, boxes, classes))
    
    #cat image to video
    input_file_name = os.path.basename(video_file)
    out_file_name = os.path.splitext(input_file_name)[0] + "_detect.mp4"
    out = os.path.join(out_dir, out_file_name)
    print(out)
    if os.path.exists(out):
        print("file %s exists." % out)
        exit(1)
    cmd = ['./scripts/synthetise_video.sh', "out", rate, str(out)]
    print(cmd)
    code = subprocess.call(cmd)
    if code == 0:
        print("video synthetise done. please check %s dir" % out_dir)
    else:
        print("something wrong, code is %d" % code)
        exit(1)
    #clear out dir
    shutil.rmtree("out")
    os.makedirs("out")
