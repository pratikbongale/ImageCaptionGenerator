import os
import pickle
import gzip
from collections import defaultdict
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

#
# loading inp documents
#

def load_doc(fname):
    '''
    Takes a file name an input and returns the file contents as a string
    :param fname: name of the file you want to read
    :return: contents of file as string
    '''

    if not os.path.exists(fname):
        print('Invalid path')
        return None

    # loads document with text descriptions into list of strings
    with open(fname, 'r') as file:
        text = file.read()

    return text

def load_photos_from_file(fname):
    '''
    :param fname: file containing list of images for training/testing/dev
    :return: a set of photo id's without format extensions
    '''
    doc = load_doc(fname)
    dataset = list()

    for line in doc.split("\n"):
        if len(line) < 1:   # line is empty
            continue

        id = line.split(".")[0]
        dataset.append(id)

    return set(dataset)     # a set of photo identifiers

def load_desc_from_file(fname, dataset):
    '''
    :param fname: file containing list of (id, img description) pairs
    :param dataset: set of img identifiers(train set/dev set/test set)
    :return: a dictionary of the form { img_id : list(image descriptions) }
             containing descriptions for image ids in provided dataset
    '''

    doc = load_doc(fname)
    desc = defaultdict(list)
    for line in doc.split('\n'):
        tokens = line.split()
        id, text = tokens[0], tokens[1:]

        if id in dataset:

            # add tokens to determine st and end of a desc
            text = 'startseq' + ' '.join(text) + 'endseq'
            desc[id].append(text)

    return desc

def load_photo_features(pkl_fname, dataset):
    """
    load the dictionary storing pre computed photo features from the pickle file
    :param fname: pickle file name
    :param dataset: set of img identifiers(train set/dev set/test set)
    :return: dictionary of form { id : list(img_features) }
    """

    with gzip.open(pkl_fname, 'rb') as pf_pickle:
        all_features = pickle.load(pf_pickle)

    features = {k : all_features[k] for k in dataset}

    return features

#
# Preparing input and output
#


def to_lines(descriptions):
    # convert a dictionary of clean descriptions to a list of descriptions
    all_desc = list()
    for id in descriptions:
        [all_desc.append(d) for d in descriptions[id]]

    return all_desc


def create_tokenizer(descriptions):
    # fit a tokenizer(maps words to numbers) to descriptions
    lines = to_lines(descriptions)

    # vectorize a text corpus, by turning each text into a sequence of integers
    # each integer being the index of a token in a dictionary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)   # updates the Tokenizer vocabulary

    return tokenizer

def max_length(descriptions):
    # calculate the length of the description with the most words
    lines = to_lines(descriptions)

    return max(len(d.split()) for d in lines)

#
# Creating input/output sequences
# Inp : X1 - arr of img features (512 for each img)
#       X2 - arr of text encoded as integers
#
# Out : y - one hot vector(binary) of the size of vocabulary
#


def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    '''
    create sequences of images, input sequences and output words for an image
    :param tokenizer: mapping for vocabulary to integers
    :param max_length: max length of a description
    :param descriptions: dictionary of descriptions {img_id : [desc1, desc2, desc3, desc4]}
    :param photots: dictionary of photo features {img_id : [f1, f2, f3 ... f512]}
    :return:
    '''

    X1, X2, y = list(), list(), list()

    for id, desc_list in descriptions.items():
        for desc in desc_list:

            seq = tokenizer.texts_to_sequences([desc])[0]   # check out Learnings.md for example

            for i in range(1, len(seq)):
                inp, out = seq[:i], seq[i]

                inp = pad_sequences([inp], maxlen=max_length)[0]   # pad 0s in front to make inp max length
                out = to_categorical([out], num_classes=vocab_size)[0]  # one hot vector of vocab_size

                X1.append(photos[id][0])
                X2.append(inp)
                y.append(out)


    return array(X1), array(X2), array(y)

#
# Sequence processor : word embedding layer + LSTM (34 words -> 256)
# Photo Feature Extractor : VGG16 (4096 elements -> 256 features)
#

def define_model(vocab_size, max_length):

    inputs1 = Input(shape=(4096,))    # create a tensor
    fe1 = Dropout(0.5)(inputs1)       # add Dropout layer to last layer in input tensor
    fe2 = Dense(256, activation='relu')(fe1)       # add dense layer to last layer in dropout

    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # decoder
    decoder1 = add([fe2, se3])      # simple sum of both layers
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # [img, seq] -> [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    return model

if __name__ == '__main__':

    #
    # Training dataset
    #

    # load list of photos to train on
    file_name = 'Dataset/Flicker8k_Tmp/trainImages.txt'     # tmp dataset
    trainset_imgs = load_photos_from_file(file_name)
    print('Dataset: %d' % len(trainset_imgs))

    # load text descriptions for images in training dataset
    file_name = 'ImgDescriptions/Flickr8k.token_tmp.txt'
    trainset_desc = load_desc_from_file(file_name, trainset_imgs)
    print('Descriptions: train=%d' % len(trainset_desc))

    # load extracted photo features for images in training dataset
    file_name = 'Features/Flicker8k_Tmp_features.pklz'
    trainset_pf = load_photo_features(file_name, trainset_imgs)
    print('Photos: train=%d' % len(trainset_pf))

    # prepare a tokenizer
    tokenizer = create_tokenizer(trainset_desc)
    vocab_size = len(tokenizer.word_index) + 1

    print('Vocabulary Size: %d' % vocab_size)

    # determine the maximum sequence length
    max_length = max_length(trainset_desc)
    print('Max description length: %d' % max_length)

    # prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, trainset_desc, trainset_pf, vocab_size)

    #
    # Dev set
    #

    # load list of photos in dev set
    file_name = 'Dataset/Flicker8k_Tmp/devImages.txt'  # tmp dataset
    devset_imgs = load_photos_from_file(file_name)
    print('Dev Dataset: %d' % len(devset_imgs))

    # load text descriptions for images in training dataset
    file_name = 'ImgDescriptions/Flickr8k.token_tmp.txt'    # contains all cleaned up desc.
    devset_desc = load_desc_from_file(file_name, devset_imgs)
    print('Descriptions: dev=%d' % len(devset_desc))

    # load extracted photo features for images in training dataset
    file_name = 'Features/Flicker8k_Tmp_features.pklz'
    devset_pf = load_photo_features(file_name, devset_imgs)
    print('Photos: dev=%d' % len(devset_pf))

    # the vocab size will be the one computed during training by the tokenizer
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, devset_desc, devset_pf, vocab_size)

    #
    # Fit model
    #

    # define the model tensors, layers, inputs and outputs
    model = define_model(vocab_size, max_length)

    # monitor the minimum loss on validation dataset(val_loss) and store to a file
    # define checkpoint callback
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))