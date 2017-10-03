#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A fun project created on Tue Oct  3 
@author: esoroush
"""

"""
Simple image visualization with TSNE dimension reduction.
This program extracts features from each image using pre-trained Networks: Inception/Vggface
using these features, TSNE algorithm draw each image in a gridded merged image.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.manifold import TSNE
import glob2

import matplotlib.pyplot as plt
from feature_extractions import Feature_extractor, supported_feature_extraction


# custom paramers: change these parameters to properly run on your machine
tf.app.flags.DEFINE_string('image_path', '/home/esoroush/Datasets/MSRC/MSRC/',
                           'Addres of all images')
tf.app.flags.DEFINE_integer('no_of_images', 1600, 'Maximum number of images')
tf.app.flags.DEFINE_boolean('stretched', False, 
                            'Determines if the resulting merged image to be stretched')
tf.app.flags.DEFINE_integer('image_width', 64, 
                            'width and height of each image in the resulting merged image')
tf.app.flags.DEFINE_string('feature_extraction', 'inception', 
                           'Determines feature extraction method. Choices are: {}'
                           .format(supported_feature_extraction))


def visualize_with_tsne(features, image_names):            
# use tsne to cluster images in 2 dimensions
    tsne = TSNE()
    reduced = tsne.fit_transform(features)
    reduced_transformed = reduced - np.min(reduced, axis=0)
    reduced_transformed /= np.max(reduced_transformed, axis=0)
    image_xindex_sorted = np.argsort(np.sum(reduced_transformed, axis=1))
    
    # draw all images in a merged image
    merged_width = int(np.ceil(np.sqrt(
            tf.flags.FLAGS.no_of_images))*tf.flags.FLAGS.image_width)
    merged_image = np.zeros((merged_width, merged_width, 3), dtype='uint8')
    
    for counter, index in enumerate(image_xindex_sorted):
        # set location
        if tf.flags.FLAGS.stretched:
            b = int(np.mod(counter, np.sqrt(tf.flags.FLAGS.no_of_images)))
            a = int(np.mod(counter//np.sqrt(tf.flags.FLAGS.no_of_images),
                           np.sqrt(tf.flags.FLAGS.no_of_images)))
            image_address = image_names[index]
            img = np.asarray(Image.open(image_address).resize(
                    (tf.flags.FLAGS.image_width, tf.flags.FLAGS.image_width)))
            merged_image[a*tf.flags.FLAGS.image_width:(a+1)*tf.flags.FLAGS.image_width,
                         b*tf.flags.FLAGS.image_width:(b+1)*tf.flags.FLAGS.image_width,
                         :] = img[:,:,:3]
        else:
            a = np.ceil(reduced_transformed[counter, 0] * 
                        (merged_width-tf.flags.FLAGS.image_width-1)+1)
            b = np.ceil(reduced_transformed[counter, 1] * (merged_width-tf.flags.FLAGS.image_width-1)+1)
            a = int(a - np.mod(a-1,tf.flags.FLAGS.image_width) + 1)
            b = int(b - np.mod(b-1,tf.flags.FLAGS.image_width) + 1)
            if merged_image[a,b,0] != 0:
                continue
            image_address = image_names[counter]
            img = np.asarray(Image.open(image_address).resize((tf.flags.FLAGS.image_width,
                             tf.flags.FLAGS.image_width)))
            merged_image[a:a+tf.flags.FLAGS.image_width, 
                         b:b+tf.flags.FLAGS.image_width,:] = img[:,:,:3]
    
    plt.imshow(merged_image)
    plt.show()
    merged_image = Image.fromarray(merged_image)
    if tf.flags.FLAGS.stretched:
        merged_image.save('merged-%s-stretched-from-%s.png'%(
                tf.flags.FLAGS.image_path.split('/')[-2], tf.flags.FLAGS.feature_extraction))
    else:
        merged_image.save('merged-%s-ellipsoide-%s.png'%(
                tf.flags.FLAGS.image_path.split('/')[-2], tf.flags.FLAGS.feature_extraction))

def main(_):
    # find all images
    image_names  = glob2.glob(tf.flags.FLAGS.image_path + "**/*.png") 
    image_names +=glob2.glob(tf.flags.FLAGS.image_path + "**/*.jpg")
    image_names +=glob2.glob(tf.flags.FLAGS.image_path + "**/*.gif")
    # suffle images
    np.random.seed(3)
    np.random.shuffle(image_names)
    if tf.flags.FLAGS.no_of_images > len(image_names):
        tf.flags.FLAGS.no_of_images = len(image_names)
    image_names = image_names[:tf.flags.FLAGS.no_of_images]
    feature_extractor = Feature_extractor(image_names)
    features = feature_extractor.extract_feature()
    visualize_with_tsne(features, image_names)
    
if __name__ == '__main__':
    tf.app.run()
    