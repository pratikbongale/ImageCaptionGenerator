# Our goal is to use transfer learning by extracting the image features using already trained VGG model
# and store them in a pickle file.
# References:
# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
# https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html

import os
import pickle
import gzip
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def extract_features(dir):

    # 1. load model
    # 2. get rid of the last layer
    # 3. extract features for each image in given dir
    # 4. store as a pickle

    #  check if directory exixts
    if not os.path.exists(dir):
        print('Unable to find images directory')
        return

    in_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))   # define the shape of inp tensor
    model = VGG16(include_top=False, input_tensor=in_layer, pooling='avg')
    model.layers.pop()

    # on first run, keras downloads and stores model weights(500 mb)
    print('Summary of the model')
    print(model.summary())

    features_dict = dict()

    for img_name in os.listdir(dir):
        fpath = os.path.join(dir, img_name)
        pil_img = load_img(fpath, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))   # load an img in Python Imaging Library(PIL) format
        img_arr = img_to_array(pil_img)     # 3D numpy array (x, y, layer(RGB))
        img_shape = img_arr.shape
        flat_img_arr = img_arr.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
        image = preprocess_input(flat_img_arr)  # preprocessing powered for imagenet
        features = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features_dict[image_id] = features
    return features_dict

def store_pickle(obj, dir, pkl_fname):
    '''
    Stores obj in directory as a compressed pickle file
    :param obj: the object to store as pickle
    :param dir: a directory path(created if one doesn't exists)
    :param pkl_fname: zipped pickle file name
    '''

    if not os.path.exists(dir):
        os.makedirs(dir)

    pkl_fname = os.path.join(dir, pkl_fname)
    with gzip.open(pkl_fname, 'wb') as pkl_file:
        pickle.dump(obj, pkl_file)


if __name__ == '__main__':

    # test 1
    dir_abs = '/home/pratik/Desktop/Projects/ImageCaptionGenerator/Dataset/Flickr8k_Dataset/Flicker8k_Dataset'
    dir_rel = 'Dataset/Flickr8k_Dataset/Flicker8k_Dataset'
    dir_tmp = 'Dataset/Flickr8k_Dataset/Flicker8k_Tmp'
    features_dict = extract_features(dir_tmp)
    assert '989851184_9ef368e520' in features_dict

    print('Extracted features for %d images' % len(features_dict))

    # store model in a pickle file
    store_pickle(obj=features_dict,
                 dir='Features',
                 pkl_fname='Flicker8k_Tmp_features.pklz')

