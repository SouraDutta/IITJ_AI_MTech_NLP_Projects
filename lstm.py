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
from keras.models import Model
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
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
data_folder = 'restaurant/category'
data_name = 'restaurant'
level = 'word'     
max_len = {'word': 79, 'char': 358}
left_max_len = {'word': 72, 'char': 344}
right_max_len = {'word': 72, 'char': 326}
asp_max_len = {'word': 21, 'char': 115}
word_embed_dim = 300
word_embed_trainable = False
word_embed_type = 'w2v'    
aspect_embed_dim = 300
aspect_embed_trainable = False
aspect_embed_type = 'w2v'     
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
exp_name = None
model_name = None
lstm_units = 300
dense_units = 128
dropout = 0.2
batch_size = 32
n_epochs = 20
learning_rate = 0.001
optimizer = "adam"
checkpoint_dir = './ckpt'
checkpoint_monitor = 'val_acc'
checkpoint_save_best_only = True
checkpoint_save_weights_only = True
checkpoint_save_weights_mode = 'max'
checkpoint_verbose = 1
early_stopping_monitor = 'val_acc'
early_stopping_patience = 5
early_stopping_verbose = 1
early_stopping_mode = 'max'
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
def load_idx2token(data_folder, vocab_type):
    vocab = load_vocab(data_folder, vocab_type)
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
    return input_data
def load_label(data_folder, data_kind):
    dirname = os.path.join('./data', data_folder)
    test_df = pd.read_csv('./data/restaurant/category/{}.csv'.format(data_kind))
    return test_df[aspect_columns].to_numpy() 
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
    content_term, aspect_term, sentiment_term, start, end = list(), list(), list(), list(), list() 
    content_cate, aspect_cate, sentiment_cate = list(), list(), list()      
    polarity = {'negative': 0, 'neutral': 1, 'positive': 2}
    global output_aspect_count, aspect_columns
    tree = ET.parse(file_path)
    print(tree)
    root = tree.getroot()
    for sentence in root:
        print(sentence)
        text = sentence.find('text').text.lower()
        for asp_cates in sentence.iter('aspectCategories'):
            for asp_cate in asp_cates.iter('aspectCategory'):
                if asp_cate.get('polarity') in polarity:
                    content_cate.append(text)
                    aspect_cate.append(asp_cate.get('category'))
                    sentiment_cate.append(polarity[asp_cate.get('polarity')])
    if not os.path.exists(os.path.join(save_folder, 'term')):
        os.makedirs(os.path.join(save_folder, 'term'))
    if len(content_cate) > 0:
        if not os.path.exists(os.path.join(save_folder, 'category')):
            os.makedirs(os.path.join(save_folder, 'category'))
        train_content, valid_content, train_aspect, valid_aspect, \
            train_senti, valid_senti = train_test_split(content_cate, aspect_cate, sentiment_cate, test_size=0.1, random_state = 42)
        train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti}
        train_data = pd.DataFrame(train_data, columns=train_data.keys())
        train_data = pd.get_dummies(train_data, columns=['aspect'], prefix='aspect')
        for ind, row in train_data.iterrows():
            print(ind, len(train_data))
            for col in train_data.columns:
                if 'aspect' in col and 'sentiment' not in col:
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
            for col in valid_data.columns:
                if 'aspect' in col and 'sentiment' not in col:
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
        output_aspect_count = len(aspect_columns)
        valid_data.drop(['sentiment'], axis = 1, inplace=True)
        valid_data = valid_data.groupby(['content'])[aspect_columns].sum().reset_index()
        valid_data.to_csv(os.path.join(save_folder, 'category/valid.csv'), index=None)
def load_glove_format(filename):
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            word_vector = np.array([float(v) for v in line[1:]])
            word_vectors[word] = word_vector
            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)
    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())
    return word_vectors, embeddings_dim
def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result
def build_vocabulary(corpus, start_id=1):
    corpus = list_flatten(corpus)
    return dict((word, idx) for idx, word in enumerate(set(corpus), start=start_id))
def build_embedding(corpus, vocab, embedding_dim=300):
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, embedding_dim), dtype='float32')
    count = 0
    for w, i in vocab.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    print('embedding out of vocabulary：', count)
    return emb
def build_glove_embedding(vocab, word_vectors, embed_dim):
    emb_matrix = np.zeros(shape=(len(vocab) + 2, embed_dim), dtype='float32')
    count = 0
    for word, i in vocab.items():
        if word not in word_vectors:
            count += 1
            emb_matrix[i, :] = np.random.uniform(-0.1, 0.1, embed_dim)
        else:
            emb_matrix[i, :] = word_vectors[word]
    print('glove embedding out of vocabulary：', count)
    return emb_matrix
def build_aspect_embedding(aspect_vocab, split_func, word_vocab, word_embed):
    aspect_embed = np.random.uniform(-0.1, 0.1, [len(aspect_vocab.keys()), word_embed.shape[1]])
    count = 0
    for aspect, aspect_id in aspect_vocab.items():
        word_ids = [word_vocab.get(word, 0) for word in split_func(aspect)]
        if any(word_ids):
            avg_vector = np.mean(word_embed[word_ids], axis=0)
            aspect_embed[aspect_id] = avg_vector
        else:
            count += 1
    print('aspect embedding out of vocabulary:', count)
    return aspect_embed
def build_aspect_text_embedding(aspect_text_vocab, word_vocab, word_embed):
    aspect_text_embed = np.zeros(shape=(len(aspect_text_vocab) + 2, word_embed.shape[1]), dtype='float32')
    count = 0
    for aspect, aspect_id in aspect_text_vocab.items():
        if aspect in word_vocab:
            aspect_text_embed[aspect_id] = word_embed[word_vocab[aspect]]
        else:
            count += 1
            aspect_text_embed[aspect_id] = np.random.uniform(-0.1, 0.1, word_embed.shape[1])
    print('aspect text embedding out of vocabulary:', count)
    return aspect_text_embed
def analyze_len_distribution(train_input, valid_input):
    text_len = list()
    text_len.extend([len(l) for l in train_input])
    text_len.extend([len(l) for l in valid_input])
    max_len = np.max(text_len)
    min_len = np.min(text_len)
    avg_len = np.average(text_len)
    median_len = np.median(text_len)
    print('max len:', max_len, 'min_len', min_len, 'avg len', avg_len, 'median len', median_len)
    for i in range(int(median_len), int(max_len), 5):
        less = list(filter(lambda x: x <= i, text_len))
        ratio = len(less) / len(text_len)
        print(i, ratio)
        if ratio >= 0.99:
            break
def analyze_class_distribution(labels):
    for cls, count in Counter(labels).most_common():
        print(cls, count, count / len(labels))
def get_loc_info(l, start, end):
    pos_info = []
    offset_info =[]
    for i in range(len(l)):
        if i < start:
            pos_info.append(1 - abs(i - start) / len(l))
            offset_info.append(i - start)
        elif start <= i < end:
            pos_info.append(1.)
            offset_info.append(0.)
        else:
            pos_info.append(1 - abs(i - end + 1) / len(l))
            offset_info.append(i - end +1)
    return pos_info, offset_info
def split_text_and_get_loc_info(data, word_vocab, char_vocab, word_cut_func):
    word_input_l, word_input_r, word_input_r_with_pad, word_pos_input, word_offset_input = [], [], [], [], []
    char_input_l, char_input_r, char_input_r_with_pad, char_pos_input, char_offset_input = [], [], [], [], []
    word_mask, char_mask = [], []
    for idx, row in data.iterrows():
        text, word_list, char_list, aspect = row['content'], row['word_list'], row['char_list'], row['aspect']
        start, end = row['from'], row['to']
        char_input_l.append(list(map(lambda x: char_vocab.get(x, len(char_vocab)+1), char_list[:end])))
        char_input_r.append(list(map(lambda x: char_vocab.get(x, len(char_vocab)+1), char_list[start:])))
        char_input_r_with_pad.append([char_vocab.get(char_list[i], len(char_vocab)+1) if i >= start else 0
                                      for i in range(len(char_list))])  
        _char_mask = [1] * len(char_list)
        _char_mask[start:end] = [0.5] * (end-start)     
        char_mask.append(_char_mask)
        _pos_input, _offset_input = get_loc_info(char_list, start, end)
        char_pos_input.append(_pos_input)
        char_offset_input.append(_offset_input)
        word_list_l = word_cut_func(text[:start])
        word_list_r = word_cut_func(text[end:])
        start = len(word_list_l)
        end = len(word_list) - len(word_list_r)
        if word_list[start:end] != word_cut_func(aspect):
            if word_list[start-1:end] == word_cut_func(aspect):
                start -= 1
            elif word_list[start:end+1] == word_cut_func(aspect):
                end += 1
            else:
                raise Exception('Can not find aspect `{}` in `{}`, word list : `{}`'.format(aspect, text, word_list))
        word_input_l.append(list(map(lambda x: word_vocab.get(x, len(word_vocab)+1), word_list[:end])))
        word_input_r.append(list(map(lambda x: word_vocab.get(x, len(word_vocab)+1), word_list[start:])))
        word_input_r_with_pad.append([word_vocab.get(word_list[i], len(word_vocab) + 1) if i >= start else 0
                                      for i in range(len(word_list))])      
        _word_mask = [1] * len(word_list)
        _word_mask[start:end] = [0.5] * (end - start)  
        word_mask.append(_word_mask)
        _pos_input, _offset_input = get_loc_info(word_list, start, end)
        word_pos_input.append(_pos_input)
        word_offset_input.append(_offset_input)
    return (word_input_l, word_input_r, word_input_r_with_pad, word_mask, word_pos_input, word_offset_input,
            char_input_l, char_input_r, char_input_r_with_pad, char_mask, char_pos_input, char_offset_input)
def pre_process(file_folder, word_cut_func, is_en):
    print('preprocessing: ', file_folder)
    train_data = pd.read_csv(os.path.join(file_folder, 'train.csv'), header=0, index_col=None)
    train_data['word_list'] = train_data['content'].apply(word_cut_func)
    train_data['char_list'] = train_data['content'].apply(lambda x: list(x))
    valid_data = pd.read_csv(os.path.join(file_folder, 'valid.csv'), header=0, index_col=None)
    valid_data['word_list'] = valid_data['content'].apply(word_cut_func)
    valid_data['char_list'] = valid_data['content'].apply(lambda x: list(x))
    print('size of training set:', len(train_data))
    print('size of valid set:', len(valid_data))
    word_corpus = np.concatenate((train_data['word_list'].values, valid_data['word_list'].values)).tolist()
    char_corpus = np.concatenate((train_data['char_list'].values, valid_data['char_list'].values)).tolist()
    print('building vocabulary...')
    word_vocab = build_vocabulary(word_corpus, start_id=1)
    char_vocab = build_vocabulary(char_corpus, start_id=1)
    pickle_dump(word_vocab, os.path.join(file_folder, 'word_vocab.pkl'))
    pickle_dump(char_vocab, os.path.join(file_folder, 'char_vocab.pkl'))
    print('finished building vocabulary!')
    print('len of word vocabulary:', len(word_vocab))
    print('sample of word vocabulary:', list(word_vocab.items())[:10])
    print('len of char vocabulary:', len(char_vocab))
    print('sample of char vocabulary:', list(char_vocab.items())[:10])
    word_embed_dim = 300
    print('preparing embedding...')
    word_w2v = build_embedding(word_corpus, word_vocab, word_embed_dim)
    char_w2v = build_embedding(char_corpus, char_vocab, word_embed_dim)
    np.save(os.path.join(file_folder, 'word_w2v.npy'), word_w2v)
    np.save(os.path.join(file_folder, 'char_w2v.npy'), char_w2v)
    print('finished preparing embedding!')
    print('shape of word_w2v:', word_w2v.shape)
    print('sample of word_w2v:', word_w2v[:2, :5])
    print('shape of char_w2v:', char_w2v.shape)
    print('sample of char_w2v:', char_w2v[:2, :5])
    print('preparing text input...')
    train_word_input = train_data['word_list'].apply(
        lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]).values.tolist()
    train_char_input = train_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    valid_word_input = valid_data['word_list'].apply(
        lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]).values.tolist()
    valid_char_input = valid_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    pickle_dump(train_word_input, os.path.join(file_folder, 'train_word_input.pkl'))
    pickle_dump(train_char_input, os.path.join(file_folder, 'train_char_input.pkl'))
    pickle_dump(valid_word_input, os.path.join(file_folder, 'valid_word_input.pkl'))
    pickle_dump(valid_char_input, os.path.join(file_folder, 'valid_char_input.pkl'))
    print('finished preparing text input!')
    print('length analysis of text word input:')
    analyze_len_distribution(train_word_input, valid_word_input)
    print('length analysis of text char input')
    analyze_len_distribution(train_char_input, valid_char_input)
    print('preparing aspect input...')
    print('finished preparing aspect input!')
    print('preparing aspect text input...')
    print('finished preparing aspect text input!')
    print('length analysis of aspect text word input:')
    print('length analysis of aspect text char input')
    if 'from' in train_data.columns:
        print('preparing left text input, right text input & position input...')
        train_word_input_l, train_word_input_r, train_word_input_r_with_pad, train_word_mask, train_word_pos_input, \
            train_word_offset_input, train_char_input_l, train_char_input_r, train_char_input_r_with_pad, \
            train_char_mask, train_char_pos_input, train_char_offset_input = split_text_and_get_loc_info(train_data,
                                                                                                         word_vocab,
                                                                                                         char_vocab,
                                                                                                         word_cut_func)
        pickle_dump(train_word_input_l, os.path.join(file_folder, 'train_word_input_l.pkl'))
        pickle_dump(train_word_input_r, os.path.join(file_folder, 'train_word_input_r.pkl'))
        pickle_dump(train_word_input_r_with_pad, os.path.join(file_folder, 'train_word_input_r_with_pad.pkl'))
        pickle_dump(train_word_mask, os.path.join(file_folder, 'train_word_mask.pkl'))
        pickle_dump(train_word_pos_input, os.path.join(file_folder, 'train_word_pos_input.pkl'))
        pickle_dump(train_word_offset_input, os.path.join(file_folder, 'train_word_offset_input.pkl'))
        pickle_dump(train_char_input_l, os.path.join(file_folder, 'train_char_input_l.pkl'))
        pickle_dump(train_char_input_r, os.path.join(file_folder, 'train_char_input_r.pkl'))
        pickle_dump(train_char_input_r_with_pad, os.path.join(file_folder, 'train_char_input_r_with_pad.pkl'))
        pickle_dump(train_char_mask, os.path.join(file_folder, 'train_char_mask.pkl'))
        pickle_dump(train_char_pos_input, os.path.join(file_folder, 'train_char_pos_input.pkl'))
        pickle_dump(train_char_offset_input, os.path.join(file_folder, 'train_char_offset_input.pkl'))
        valid_word_input_l, valid_word_input_r, valid_word_input_r_with_pad, valid_word_mask, valid_word_pos_input, \
            valid_word_offset_input, valid_char_input_l, valid_char_input_r, valid_char_input_r_with_pad, \
            valid_char_mask, valid_char_pos_input, valid_char_offset_input = split_text_and_get_loc_info(valid_data,
                                                                                                         word_vocab,
                                                                                                         char_vocab,
                                                                                                         word_cut_func)
        pickle_dump(valid_word_input_l, os.path.join(file_folder, 'valid_word_input_l.pkl'))
        pickle_dump(valid_word_input_r, os.path.join(file_folder, 'valid_word_input_r.pkl'))
        pickle_dump(valid_word_input_r_with_pad, os.path.join(file_folder, 'valid_word_input_r_with_pad.pkl'))
        pickle_dump(valid_word_mask, os.path.join(file_folder, 'valid_word_mask.pkl'))
        pickle_dump(valid_word_pos_input, os.path.join(file_folder, 'valid_word_pos_input.pkl'))
        pickle_dump(valid_word_offset_input, os.path.join(file_folder, 'valid_word_offset_input.pkl'))
        pickle_dump(valid_char_input_l, os.path.join(file_folder, 'valid_char_input_l.pkl'))
        pickle_dump(valid_char_input_r, os.path.join(file_folder, 'valid_char_input_r.pkl'))
        pickle_dump(valid_char_input_r_with_pad, os.path.join(file_folder, 'valid_char_input_r_with_pad.pkl'))
        pickle_dump(valid_char_mask, os.path.join(file_folder, 'valid_char_mask.pkl'))
        pickle_dump(valid_char_pos_input, os.path.join(file_folder, 'valid_char_pos_input.pkl'))
        pickle_dump(valid_char_offset_input, os.path.join(file_folder, 'valid_char_offset_input.pkl'))
        print('length analysis of left text word input:')
        analyze_len_distribution(train_word_input_l, valid_word_input_l)
        print('length analysis of left text char input')
        analyze_len_distribution(train_char_input_l, valid_char_input_l)
        print('length analysis of right text word input:')
        analyze_len_distribution(train_word_input_r, valid_word_input_r)
        print('length analysis of right text char input')
        analyze_len_distribution(train_char_input_r, valid_char_input_r)
    print('preparing output....')
    print('finished preparing output!')
    print('class analysis of training set:')
    print('class analysis of valid set:')
process_xml('train.xml', is_train_file=True,
                save_folder='./data/restaurant')
pre_process('./data/restaurant/category', lambda x: nltk.word_tokenize(x), True)
def ae_lstm():
    input_text = Input(shape=(max_len,))
    input_aspect = Input(shape=(output_aspect_count,),)
    word_embedding = Embedding(input_dim=text_embeddings.shape[0], output_dim=word_embed_dim,
                               weights=[text_embeddings], trainable=word_embed_trainable,
                               mask_zero=True)
    text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))
    hidden = LSTM(lstm_units)(text_embed)
    return Model([input_text], hidden)
def build_base_network():
    base_network = ae_lstm()
    return base_network
model = None
def build_model():
    global model
    network_inputs = list()
    if use_text_input:
        network_inputs.append(Input(shape=(max_len,), name='input_text'))
    if use_text_input_l:
        network_inputs.append(Input(shape=(left_max_len,), name='input_text_l'))
    if use_text_input_r:
        network_inputs.append(Input(shape=(right_max_len,), name='input_text_r'))
    if use_text_input_r_with_pad:
        network_inputs.append(Input(shape=(max_len,), name='input_text_r_with_pad'))
    if use_aspect_input:
        network_inputs.append(Input(shape=(1, ), name='input_aspect'))
    if use_aspect_text_input:
        network_inputs.append(Input(shape=(asp_max_len,), name='input_aspect_text'))
    if use_loc_input:
        network_inputs.append(Input(shape=(max_len,), name='input_loc_info'))
    if use_offset_input:
        network_inputs.append(Input(shape=(max_len,), name='input_offset_info'))
    if use_mask:
        network_inputs.append(Input(shape=(max_len,), name='input_mask'))
    if len(network_inputs) == 1:
        network_inputs = network_inputs[0]
    elif len(network_inputs) == 0:
        raise Exception('No Input!')
    base_network = build_base_network()
    sentence_vec = base_network(network_inputs)
    dense_layer = Dense(dense_units, activation='relu')(sentence_vec)
    output_layer = Dense(output_aspect_count, activation='sigmoid')(dense_layer)
    model = Model(network_inputs, output_layer)
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)
def prepare_input(model_name, input_data):
    if model_name in ['at_lstm', 'ae_lstm', 'atae_lstm'] or \
            (model_name in ['memnet', 'ram'] and not is_aspect_term):
        text = input_data
        input_pad = pad_sequences(text, max_len, dtype='float32')
    return input_pad
def prepare_label(label_data):
    print(n_classes)
    return to_categorical(label_data, n_classes)
def train(model_name, train_input_data, train_label, valid_input_data, valid_label):
    global model
    x_train = prepare_input(model_name, train_input_data)
    y_train = train_label 
    x_valid = prepare_input(model_name, valid_input_data)
    y_valid = valid_label 
    x_train = np.asarray(x_train).astype('float32')
    x_valid = np.asarray(x_valid).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_valid = np.asarray(y_valid).astype('float32')
    print('start training...')
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs,
                   validation_data=(x_valid, y_valid))
    print('training end...')
    model.save('lstm_model')
    print('score over valid data:')
    valid_pred = predict(model_name, x_valid)
    get_score_senti(y_valid, valid_pred)
def predict(model_name, input_data):
    global model
    input_pad = prepare_input(model_name,input_data)
    prediction = model.predict(input_pad)
    print(prediction)
    return np.argmax(prediction, axis=-1)
def train_model(data_folder, data_name, level, model_name, is_aspect_term=True):
    if not os.path.exists(os.path.join(checkpoint_dir, data_folder)):
        os.makedirs(os.path.join(checkpoint_dir, data_folder))
    exp_name = '{}_{}_wv_{}'.format(model_name, level, word_embed_type)
    exp_name = exp_name + '_update' if word_embed_trainable else exp_name + '_fix'
    if use_aspect_input:
        exp_name += '_aspv_{}'.format(aspect_embed_type)
        exp_name = exp_name + '_update' if aspect_embed_trainable else exp_name + '_fix'
    if use_elmo:
        exp_name += '_elmo_alone_{}_mode_{}_{}'.format(use_elmo_alone, elmo_output_mode,
                                                              'update' if elmo_trainable else 'fix')
    print(exp_name)
    if not os.path.exists(os.path.join(checkpoint_dir, '%s/%s.hdf5' % (data_folder, exp_name))):
        start_time = time.time()
        train_input = load_input_data(data_folder, 'train', level, use_text_input, use_text_input_l,
                                      use_text_input_r, use_text_input_r_with_pad,
                                      use_aspect_input, use_aspect_text_input, use_loc_input,
                                      use_offset_input, use_mask)
        train_label = load_label(data_folder, 'train')
        valid_input = load_input_data(data_folder, 'valid', level, use_text_input, use_text_input_l,
                                      use_text_input_r, use_text_input_r_with_pad,
                                      use_aspect_input, use_aspect_text_input, use_loc_input,
                                      use_offset_input, use_mask)
        valid_label = load_label(data_folder, 'valid')
        print(valid_label)
        print(train_label)
        
        train_combine_valid_input = []
        train(model_name, train_input, train_label, valid_input, valid_label)
        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
if __name__ == '__main__':
    use_elmo = False
    use_elmo_alone = False
    elmo_trainable = False
    word_embed_trainable = True
    aspect_embed_trainable = True
    max_len = max_len[level]
    left_max_len = left_max_len[level]
    right_max_len = right_max_len[level]
    asp_max_len = asp_max_len[level]
    if use_text_input or use_text_input_l or use_text_input_r or use_text_input_r_with_pad:
        text_embeddings = np.load('./data/%s/%s_%s.npy' % (data_folder, level,
                                                                word_embed_type))
        idx2token = load_idx2token(data_folder, level)
    else:
        text_embeddings = None
    if use_aspect_input:
        aspect_embeddings = np.load('./data/%s/aspect_%s_%s.npy' % (data_folder, level,
                                                                         aspect_embed_type))
        if aspect_embed_type == 'random':
            n_aspect = aspect_embeddings.shape[0]
            aspect_embeddings = None
    else:
        aspect_embeddings = None
    if use_aspect_text_input:
        aspect_text_embeddings = np.load('./data/%s/aspect_text_%s_%s.npy' % (data_folder,
                                                                                   level,
                                                                                   word_embed_type))
        idx2aspect_token = load_idx2token(data_folder, 'aspect_text_{}'.format(level))
    else:
        aspect_text_embeddings = None
    model = None
    build_model()
    train_model('restaurant/category', 'restaurant', 'word', 'ae_lstm')
