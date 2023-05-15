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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
data_folder = 'restaurant/category'
data_name = 'restaurant'
level = 'word'     
max_len = {'word': 72, 'char': 358}
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
target_senti = None
exp_name = None
model_name = None
lstm_units = 300
dense_units = 128
dropout = 0.2
batch_size = 32
n_epochs = 5
n_classes = 3
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
aspect_columns = ['aspect_ambience', 'aspect_food', 'aspect_menu', 'aspect_miscellaneous', 'aspect_place', 'aspect_price', 'aspect_service', 'aspect_staff', 'aspect_ambience_sentiment_0', 'aspect_ambience_sentiment_1', 'aspect_ambience_sentiment_2', 'aspect_food_sentiment_0', 'aspect_food_sentiment_1', 'aspect_food_sentiment_2', 'aspect_menu_sentiment_0', 'aspect_menu_sentiment_1', 'aspect_menu_sentiment_2', 'aspect_miscellaneous_sentiment_0', 'aspect_miscellaneous_sentiment_1', 'aspect_miscellaneous_sentiment_2', 'aspect_place_sentiment_0', 'aspect_place_sentiment_1', 'aspect_place_sentiment_2', 'aspect_price_sentiment_0', 'aspect_price_sentiment_1', 'aspect_price_sentiment_2', 'aspect_service_sentiment_0', 'aspect_service_sentiment_1', 'aspect_service_sentiment_2', 'aspect_staff_sentiment_0', 'aspect_staff_sentiment_1', 'aspect_staff_sentiment_2']
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
    print(input_data)
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
def process_xml(file_path, is_train_file, save_folder, max_len):
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
        content_cate.append(text)
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
        # if not is_train_file:
        #     test_data = {'content': content_cate, 'aspect': aspect_cate, 'sentiment': sentiment_cate}
        #     test_data = pd.DataFrame(test_data, columns=test_data.keys())
        #     test_data.to_csv(os.path.join(save_folder, 'category/test.csv'), index=None)
        # else:
        valid_data = {'content': content_cate}
        valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
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
def analyze_len_distribution(valid_input):
    text_len = list()
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
        text, word_list, char_list = row['content'], row['word_list'], row['char_list']
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
    valid_data = pd.read_csv(os.path.join(file_folder, 'valid.csv'), header=0, index_col=None)
    valid_data['word_list'] = valid_data['content'].apply(word_cut_func)
    valid_data['char_list'] = valid_data['content'].apply(lambda x: list(x))
    print('size of valid set:', len(valid_data))
    valid_data['word_list'] = valid_data['word_list'].apply(lambda x: x[0:max_len])
    word_corpus = (valid_data['word_list'].values).tolist()
    char_corpus = (valid_data['char_list'].values).tolist()
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
    word_embed_dim = 1000
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
    valid_word_input = valid_data['word_list'].apply(
        lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]).values.tolist()
    valid_char_input = valid_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    pickle_dump(valid_word_input, os.path.join(file_folder, 'valid_word_input.pkl'))
    pickle_dump(valid_char_input, os.path.join(file_folder, 'valid_char_input.pkl'))
    print('finished preparing text input!')
    print('length analysis of text word input:')
    analyze_len_distribution(valid_word_input)
    print('length analysis of text char input')
    analyze_len_distribution(valid_char_input)
    print('preparing aspect input...')
    print('finished preparing aspect input!')
    print('preparing aspect text input...')
    print('finished preparing aspect text input!')
    print('length analysis of aspect text word input:')
    print('length analysis of aspect text char input')
    print('preparing output....')
    print('finished preparing output!')
    print('class analysis of training set:')
    print('class analysis of valid set:')
def prepare_input(model_name, input_data):
    if model_name in ['at_lstm', 'ae_lstm', 'atae_lstm'] or \
            (model_name in ['memnet', 'ram'] and not is_aspect_term):
        text = input_data
        input_pad = pad_sequences(text, max_len, dtype='float32')
    return input_pad
def prepare_label(label_data):
    print(n_classes)
    return to_categorical(label_data, n_classes)
def run(data_folder, data_name, level, model_name, is_aspect_term=True):
    global model
    valid_input = load_input_data(data_folder, 'valid', level, use_text_input, use_text_input_l,
                                  use_text_input_r, use_text_input_r_with_pad,
                                  use_aspect_input, use_aspect_text_input, use_loc_input,
                                  use_offset_input, use_mask)
    x_valid = prepare_input(model_name, valid_input)
    x_valid = np.asarray(x_valid).astype('float32')
    model = load_model('lstm_model')
    print('score over valid data:')
    print(x_valid.shape)
    valid_pred = model.predict(x_valid)
    valid_df = pd.DataFrame(valid_pred, columns = aspect_columns)
    print(valid_df)
    valid_df.to_csv('output_lstm.csv', index=False)
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
    process_xml('test.xml', is_train_file=True,
                save_folder='./data/restaurant', max_len=max_len)
    pre_process('./data/restaurant/category', lambda x: nltk.word_tokenize(x), True)
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
    run('restaurant/category', 'restaurant', 'word', 'ae_lstm')
