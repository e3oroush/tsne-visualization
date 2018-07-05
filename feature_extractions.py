#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3
This is auxilary class for tsne visualization project.
This class objective is extracting features from images.

@author: esoroush
"""
import os, pickle

from keras.engine import  Model
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils

import numpy as np
import tensorflow as tf
from PIL import Image

supported_feature_extraction = ['inception', 'raw', 'vggfaces']
# TODO: This class should be standalone and independent with tf.app.flags
# Google inception pre-trained network
class FeatureExtractor(object):
    def __init__(self, image_names):
        self.image_names = image_names
        self.no_of_images = len(image_names)
        self.image_path = image_names[0].split('/')[-2]
        self.supported_feature_extraction = supported_feature_extraction
    def extract_feature(self):
        if tf.flags.FLAGS.feature_extraction == 'raw':
            features = self.raw_features()
        elif tf.flags.FLAGS.feature_extraction == 'inception':
            features = self.inception_features()
        elif tf.flags.FLAGS.feature_extraction == 'vggfaces':
            features = self.vggfaces_features()
        else:
            print('Currently the supported methods for feature extraction are: {}'.format(self.supported_feature_extraction))
            return
        return features
    def inception_features(self):
        print('using Inception network for feature extraction')
        import sys, tarfile
        from six.moves import urllib
        model_dir = os.path.join(os.environ['HOME'], '.tensorflow/models')
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        def create_graph():
    
            """Creates a graph from saved GraphDef file and returns a saver."""
            # Creates graph from saved graph_def.pb.
            with tf.gfile.FastGFile(os.path.join(
              model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
    
        def run_inference_on_image(image):
    
            """Runs forward path on an image.
            Args:
            image: Image file name.
    
            Returns:
            off the shelf 2048 feature vector
            """
            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = tf.gfile.FastGFile(image, 'rb').read()
    
    
            with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
                pool3 = sess.graph.get_tensor_by_name('pool_3:0')
                features = sess.run(pool3,
                                       {'DecodeJpeg/contents:0': image_data})
                return features
        
        def maybe_download_and_extract():
            """Download and extract model tar file."""
            dest_directory = model_dir
            if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)
            filename = DATA_URL.split('/')[-1]
            filepath = os.path.join(dest_directory, filename)
            if not os.path.exists(filepath):
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % ( 
                          filename, float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()
                filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
                print()
                statinfo = os.stat(filepath)
                print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
    
        maybe_download_and_extract()
        # Creates graph from saved GraphDef.
        create_graph()
        feature_filename = '%s-feature-inception-%d.p'%(
                self.image_path, self.no_of_images)
        if os.path.exists(feature_filename):
            with open(feature_filename, 'rb') as f:
                features, self.image_names = pickle.load(f)
        else:
            features = np.zeros([self.no_of_images, 2048])
            for i in range(self.no_of_images):
                print('image name: %s index: %d/%d' %(
                        self.image_names[i], i, self.no_of_images))
                features[i, :] = run_inference_on_image(image=self.image_names[i]).squeeze()
            with open(feature_filename, 'wb') as f:
                pickle.dump((features, self.image_names), f)
        return features
                
    # raw image pixels resized to 100x100
    def raw_features(self):
        print('using raw method for feature extraction')
        features = np.zeros([self.no_of_images, 100*100])
        for i, name in enumerate(self.image_names):
            features[i, :] = np.asarray(Image.open(name).resize((100, 100)).
                    convert('L')).reshape(-1,)
            
    
    # vgg face pretrained network
    def vggfaces_features(self):
        print('using vggfaces network for feature extraction')
        # Convolution Features
        features = np.zeros([self.no_of_images, 2048])
    #    vgg_model_conv = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max
        # FC7 Features
        vgg_model = VGGFace() # pooling: None, avg or max
        out = vgg_model.get_layer('fc7').output
        vgg_model_fc7 = Model(vgg_model.input, out)
    
        feature_filename = '%s-feature-vggfaces-%d.p'%(
                self.image_path, self.no_of_images)
        if os.path.exists(feature_filename):
            with open(feature_filename, 'rb') as f:
                features, self.image_names = pickle.load(f)
        else:
            features = np.zeros([self.no_of_images, 4096])
            for i, name in enumerate(self.image_names):
                img = image.load_img(name, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x)
                print('image name: %s progress: %d/%d'%(
                        name, i+1, self.no_of_images))
                features[i, :] = vgg_model_fc7.predict(x)
            with open(feature_filename, 'wb') as f:
                pickle.dump((features, self.image_names), f)
        return features