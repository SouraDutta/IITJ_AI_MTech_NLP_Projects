import os
import codecs
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from sklearn.metrics import f1_score, accuracy_score
import time
from keras.models import Model, load_model
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, Conv1D, MaxPool1D, Flatten, concatenate, Dense, \
    LSTM, Bidirectional, Activation, MaxPooling1D, Add, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    TimeDistributed, Permute, multiply, Lambda, add, Masking, BatchNormalization, Softmax, Reshape, ReLU, \
    ZeroPadding1D, subtract
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import keras.backend as K
import tensorflow as tf
import warnings
import tensorflow_hub as hub
import tokenization

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = "1"
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True, name='bert_layer')

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

data_folder = 'restaurant/category'
data_name = 'restaurant'
level = 'word'     # options are 'word' & 'char'
max_len = {'word': 79, 'char': 358}
left_max_len = {'word': 72, 'char': 344}
right_max_len = {'word': 72, 'char': 326}
asp_max_len = {'word': 21, 'char': 115}
word_embed_dim = 300
word_embed_trainable = False
word_embed_type = 'w2v'    # use what kind of word embeddings, can be pre-trained on a larger corpus or just on dataset
aspect_embed_dim = 300
aspect_embed_trainable = False
aspect_embed_type = 'w2v'     # use mean of word embeddings or just randomly initialization
use_text_input = None
use_text_input_l = None
use_text_input_r = None
use_text_input_r_with_pad = None
use_aspect_input = None
use_aspect_text_input = None
use_loc_input = None
use_offset_input = None
use_mask = None
is_aspect_term = True

# model structure configuration
exp_name = None
model_name = None
lstm_units = 300
dense_units = 128

# model training configuration
dropout = 0.2
batch_size = 128
n_epochs = 3
n_classes = 3
learning_rate = 0.001
optimizer = "adam"

# model saving configuration
checkpoint_dir = './ckpt'
checkpoint_monitor = 'val_acc'
checkpoint_save_best_only = True
checkpoint_save_weights_only = True
checkpoint_save_weights_mode = 'max'
checkpoint_verbose = 1

# early stopping configuration
early_stopping_monitor = 'val_acc'
early_stopping_patience = 5
early_stopping_verbose = 1
early_stopping_mode = 'max'

# elmo embedding configure
use_elmo = False
use_elmo_alone = False
elmo_hub_url = './raw_data/tfhub_elmo_2'
elmo_output_mode = 'elmo'
idx2token = None
idx2aspect_token = None
elmo_trainable = False
use_text_input, use_text_input_l, use_text_input_r = True, False, False
use_aspect_input, use_aspect_text_input = False, False
use_loc_input, use_offset_input = False, False

output_aspect_count = 0
aspect_columns = []

def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))


def load_input_data(data_folder, data_kind, level, use_text_input, use_text_input_l, use_text_input_r,
                    use_text_input_r_with_pad, use_aspect_input, use_aspect_text_input, use_loc_input,
                    use_offset_input, use_mask):
    dirname = os.path.join('./data', data_folder)
    input_data = []
    if use_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input.pkl'.format(data_kind, level))))
    if use_text_input_l:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_l.pkl'.format(data_kind, level))))
    if use_text_input_r:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_r.pkl'.format(data_kind, level))))
    if use_text_input_r_with_pad:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_r_with_pad.pkl'.format(data_kind, level))))
    if use_aspect_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_aspect_input.pkl'.format(data_kind))))
    if use_aspect_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_aspect_input.pkl'.format(data_kind, level))))
    if use_loc_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_pos_input.pkl'.format(data_kind, level))))
    if use_offset_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_offset_input.pkl'.format(data_kind, level))))
    if use_mask:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_mask.pkl'.format(data_kind, level))))
    if len(input_data) == 1:
        input_data = input_data[0]
    if len(input_data) == 0:
        raise Exception('No Input!')
    # print(input_data)
    return input_data


def load_label(data_folder, data_kind):
    dirname = os.path.join('./data', data_folder)

    test_df = pd.read_csv('./data/restaurant/category/{}.csv'.format(data_kind))
    return test_df[aspect_columns].to_numpy() #pickle_load(os.path.join(dirname, '{}_label.pkl'.format(data_kind)))


def load_vocab(data_folder, vocab_type):
    dirname = os.path.join('./data', data_folder)
    return pickle_load(os.path.join(dirname, vocab_type+'_vocab.pkl'))

def get_score_senti(y_true, y_pred):
    """
    return score for predictions made by sentiment analysis model
    :param y_true: array shaped [batch_size, 3]
    :param y_pred: array shaped [batch_size, 3]
    :return:
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print('acc:', acc)
    print('macro_f1:', f1)
    return acc, f1

def process_xml(file_path, is_train_file, save_folder):
    content_term, aspect_term, sentiment_term, start, end = list(), list(), list(), list(), list() # data for aspect term
    content_cate, aspect_cate, sentiment_cate = list(), list(), list()      # data for aspect category
    polarity = {'negative': 0, 'neutral': 1, 'positive': 2}
    global output_aspect_count, aspect_columns
    tree = ET.parse(file_path)
    print(tree)
    root = tree.getroot()
    for sentence in root:
        print(sentence)
        text = sentence.find('text').text.lower()
        # for asp_terms in sentence.iter('aspectCategories'):
        #     for asp_term in asp_terms.iter('aspectCategory'):
        #         if asp_term.get('polarity') in polarity:
        #             _text = text
        #             _aspect = asp_term.get('category').lower()
        #             _sentiment = polarity[asp_term.get('polarity')]
        #             # if _start > 0 and text[_start - 1] != ' ':
        #             #     _text = text[:_start] + ' ' + text[_start:]
        #             #     _start += 1
        #             #     _end += 1
        #             # if _end < len(_text) and _text[_end] != ' ':
        #             #     _text = _text[:_end] + ' ' + _text[_end:]
        #             # if _text[_start:_end] != _aspect:
        #             #     raise Exception('{}=={}=={}'.format(_text, _text[_start:_end], _aspect))
        #             content_term.append(text)
        #             aspect_term.append(_aspect)
        #             print(aspect_term)
        #             sentiment_term.append(_sentiment)
        #             # start.append(_start)
        #             # end.append(_end)
        for asp_cates in sentence.iter('aspectCategories'):
            for asp_cate in asp_cates.iter('aspectCategory'):
                if asp_cate.get('polarity') in polarity:
                    content_cate.append(text)
                    aspect_cate.append(asp_cate.get('category'))
                    sentiment_cate.append(polarity[asp_cate.get('polarity')])

    if not os.path.exists(os.path.join(save_folder, 'term')):
        os.makedirs(os.path.join(save_folder, 'term'))

    # if not is_train_file:
    #     test_data = {'content': content_term, 'aspect': aspect_term, 'sentiment': sentiment_term,
    #                  'from': start, 'to': end}
    #     test_data = pd.DataFrame(test_data, columns=test_data.keys())
    #     test_data.to_csv(os.path.join(save_folder, 'term/test.csv'), index=None)
    # else:
    # print(aspect_term)
    # train_content, valid_content, train_aspect, valid_aspect, train_senti, valid_senti, train_start, valid_start, \
    #     train_end, valid_end = train_test_split(content_term, aspect_term, sentiment_term, start, end, test_size=0.1)
    # train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti,
    #               'from': train_start, 'to': train_end}
    # train_data = pd.DataFrame(train_data, columns=train_data.keys())
    # train_data.to_csv(os.path.join(save_folder, 'term/train.csv'), index=None)
    # valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti,
    #               'from': valid_start, 'to': valid_end}
    # valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
    # valid_data.to_csv(os.path.join(save_folder, 'term/valid.csv'), index=None)

    if len(content_cate) > 0:
        if not os.path.exists(os.path.join(save_folder, 'category')):
            os.makedirs(os.path.join(save_folder, 'category'))

        train_all_tokens = []
        train_all_masks = []
        train_all_segments = []
        valid_all_tokens = []
        valid_all_masks = []
        valid_all_segments = []

        train_content, valid_content, train_aspect, valid_aspect, \
            train_senti, valid_senti = train_test_split(content_cate, aspect_cate, sentiment_cate, test_size=0.1, random_state = 42)
        train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti}
        train_data = pd.DataFrame(train_data, columns=train_data.keys())
        train_data = pd.get_dummies(train_data, columns=['aspect'], prefix='aspect')
        for ind, row in train_data.iterrows():
            print(ind, len(train_data))
            text = tokenizer.tokenize(train_data.loc[ind, 'content'])
            
            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)
            
            tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len
            
            if not tokens in train_all_tokens:
                train_all_tokens.append(tokens)
                train_all_masks.append(pad_masks)
                train_all_segments.append(segment_ids)
                            

            for col in train_data.columns:
                if 'aspect' in col and 'sentiment' not in col:
                    # print(col)
                    if (col+'_sentiment_0' not in train_data.columns):
                        train_data[col+'_sentiment_0'] = 0
                    
                    if (col+'_sentiment_1' not in train_data.columns):
                        train_data[col+'_sentiment_1'] = 0
                    
                    if (col+'_sentiment_2' not in train_data.columns):
                        train_data[col+'_sentiment_2'] = 0
                    
                    train_data.loc[ind, col+'_sentiment_0'] = 1 if train_data.loc[ind, 'sentiment'] == 0 and train_data.loc[ind, col] == 1 else 0
                    train_data.loc[ind, col+'_sentiment_1'] = 1 if train_data.loc[ind, 'sentiment'] == 1 and train_data.loc[ind, col] == 1 else 0
                    train_data.loc[ind, col+'_sentiment_2'] = 1 if train_data.loc[ind, 'sentiment'] == 2 and train_data.loc[ind, col] == 1 else 0
        aspect_columns = [col for col in train_data.columns if ('aspect' in col)]
        # print(asp_cols)
        output_aspect_count = len(aspect_columns)
        train_data.drop(['sentiment'], axis = 1, inplace=True)
        train_data = train_data.groupby(['content'])[aspect_columns].sum().reset_index()

        print(train_data)
        train_data.to_csv(os.path.join(save_folder, 'category/train.csv'), index=None)
        valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti}
        valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())

        valid_data = pd.get_dummies(valid_data, columns=['aspect'], prefix='aspect')
        for ind, row in valid_data.iterrows():
            print(ind, len(valid_data))

            text = tokenizer.tokenize(valid_data.loc[ind, 'content'])
            
            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)
            
            tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len
            
            if not tokens in valid_all_tokens:
                valid_all_tokens.append(tokens)
                valid_all_masks.append(pad_masks)
                valid_all_segments.append(segment_ids)
            
            for col in valid_data.columns:
                if 'aspect' in col and 'sentiment' not in col:
                    # print(col)
                    if (col+'_sentiment_0' not in valid_data.columns):
                        valid_data[col+'_sentiment_0'] = 0
                    
                    if (col+'_sentiment_1' not in valid_data.columns):
                        valid_data[col+'_sentiment_1'] = 0
                    
                    if (col+'_sentiment_2' not in valid_data.columns):
                        valid_data[col+'_sentiment_2'] = 0
                    
                    valid_data.loc[ind, col+'_sentiment_0'] = 1 if valid_data.loc[ind, 'sentiment'] == 0 and valid_data.loc[ind, col] == 1 else 0
                    valid_data.loc[ind, col+'_sentiment_1'] = 1 if valid_data.loc[ind, 'sentiment'] == 1 and valid_data.loc[ind, col] == 1 else 0
                    valid_data.loc[ind, col+'_sentiment_2'] = 1 if valid_data.loc[ind, 'sentiment'] == 2 and valid_data.loc[ind, col] == 1 else 0
        aspect_columns = [col for col in valid_data.columns if ('aspect' in col)]
        # print(asp_cols)
        output_aspect_count = len(aspect_columns)

        valid_data.drop(['sentiment'], axis = 1, inplace=True)
        valid_data = valid_data.groupby(['content'])[aspect_columns].sum().reset_index()
        valid_data.to_csv(os.path.join(save_folder, 'category/valid.csv'), index=None)

    return np.array(train_all_tokens), np.array(train_all_masks), np.array(train_all_segments), np.array(valid_all_tokens), np.array(valid_all_masks), np.array(valid_all_segments)

# config = Config()
# glove_vectors, glove_embed_dim = load_glove_format('./raw_data/glove.42B.300d.txt')

model = None

def build_model(bert_layer, max_len=512):
    
    global model

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = Dense(64, activation='relu')(clf_output)
    net = Dropout(0.4)(net)
    out = Dense(32, activation='sigmoid')(net)
    # net = Dropout(0.2)(net)
    # out = Dense(5, activation='softmax')(net)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def run(model_name, train_all_tokens, train_all_masks, train_all_segments, train_label, valid_all_tokens, valid_all_masks, valid_all_segments, valid_label):
    
    global model 

    model = load_model('bert_model')
    
    y_valid = np.asarray(valid_label).astype('float32')

    print('score over valid data:')
    print((valid_all_tokens, valid_all_masks, valid_all_segments))
    valid_pred = model.predict((valid_all_tokens, valid_all_masks, valid_all_segments))
    valid_df = pd.DataFrame(valid_pred, columns = aspect_columns)

    print(valid_df)
    get_score_senti(y_valid, valid_pred)

# def score(input_data, label):
#     global model
#     input_pad = prepare_input(model_name, input_data)
#     label = prepare_label(label)
#     prediction = model.predict(input_pad)
#     get_score_senti(label, prediction)


def run_model(data_folder, data_name, level, model_name, train_all_tokens, train_all_masks, train_all_segments, valid_all_tokens, valid_all_masks, valid_all_segments, is_aspect_term=True):
    # data_folder = data_folder
    # data_name = data_name
   

    # test_input = load_input_data(data_folder, 'test', level, use_text_input, use_text_input_l,
    #                              use_text_input_r, use_text_input_r_with_pad, use_aspect_input,
    #                              use_aspect_text_input, use_loc_input, use_offset_input,
    #                              use_mask)
    # test_label = load_label(data_folder, 'test')

    if not os.path.exists(os.path.join(checkpoint_dir, '%s/%s.hdf5' % (data_folder, exp_name))):
        start_time = time.time()

        
        train_label = load_label(data_folder, 'train')
        
        valid_label = load_label(data_folder, 'valid')
        print(valid_label)
        print(train_label)

        '''
        Note: Here I combine the training data and validation data together, use them as training input to the model, 
              while I use test data to server as validation input. The reason behind is that i want to fully explore how 
              well can the model perform on the test data (Keras's ModelCheckpoint callback can help usesave the model 
              which perform best on validation data (here the test data)).
              But generally, we won't do that, because test data will not (and should not) be accessible during training 
              process.
        '''
        train_combine_valid_input = []
        # for i in range(len(train_input)):
        #     train_combine_valid_input.append(train_input[i] + valid_input[i])
        # train_combine_valid_label = train_label + valid_label

        run(model_name, train_all_tokens, train_all_masks, train_all_segments, train_label, valid_all_tokens, valid_all_masks, valid_all_segments, valid_label)

        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # load the best model
    # model.load()

    # print('score over valid data...')
    # model.score(valid_input, valid_label)
    


if __name__ == '__main__':
    # config = Config()
    use_elmo = False
    use_elmo_alone = False
    elmo_trainable = False

    word_embed_trainable = True
    aspect_embed_trainable = True
    
    # level = level
    # use_elmo = use_elmo
    max_len = max_len[level]
    left_max_len = left_max_len[level]
    right_max_len = right_max_len[level]
    asp_max_len = asp_max_len[level]

    train_all_tokens, train_all_masks, train_all_segments, valid_all_tokens, valid_all_masks, valid_all_segments = process_xml('train.xml', is_train_file=True,
                save_folder='./data/restaurant')

    # build_model()

    run_model('restaurant/category', 'restaurant', 'word', 'ae_lstm', train_all_tokens, train_all_masks, train_all_segments, valid_all_tokens, valid_all_masks, valid_all_segments)

    # word_embed_trainable = False
    # aspect_embed_trainable = True
 
    # train_model('restaurant/category', 'restaurant', 'word', 'ae_lstm')

    # word_embed_trainable = False
    # aspect_embed_trainable = False

    # train_model('restaurant/category', 'restaurant', 'word', 'ae_lstm')


