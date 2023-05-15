from DataLoader import Loader
import os
import config
import model
import json
from Rel_based_labeling import Rel_based_labeling
import torch
import numpy as np
import nltk

opt = config.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

def load_label(input_label2id):
    label2id = json.load(open(input_label2id, 'r'))
    return label2id

def is_normal_triple(triples):
    """
    normal triples means triples are not over lap in entity.
    example [[e1,e2,r1], [e3,e4,r2]]
    :param triples
    :param is_relation_first
    :return:

    >>> is_normal_triple([1,2,3, 4,5,0])
    True
    >>> is_normal_triple([1,2,3, 4,5,3])
    True
    >>> is_normal_triple([1,2,3, 2,5,0])
    False
    >>> is_normal_triple([1,2,3, 1,2,0])
    False
    """
    entities = set()
    for e in triples:
        entities.add(e[0])
        entities.add(e[1])
    if len(entities) != 2 * len(triples):
        return False
    return True

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    #tag_type = content[1]
    if len(content) == 1:
        return tag_class
    ht = content[-1]
    return tag_class, ht


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    #print(seq)
    #print(tags)
    default1 = tags['O']
    default2 = tags['X']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if (tok == default1 or tok == default2) and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1 and tok != default2:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def attn_mapping(attn_scores, gts):
    gt_rel = []
    for gt in gts:
        gt_rel.append(gt[2] - 1)
    return attn_scores[gt_rel]

def tag_mapping(predict_tags, cur_relation, label2id):
    '''
    parameters
        predict_tags : np.array, shape: (rel_number, max_sen_len)
        cur_relation : list of relation id
    '''
    assert predict_tags.shape[0] == len(cur_relation)

    predict_triples = []
    for i in range(predict_tags.shape[0]):
        heads = []
        tails = []
        pred_chunks = get_chunks(predict_tags[i], label2id)
        for ch in pred_chunks:
            if ch[0].split('-')[-1] == 'H':
                heads.append(ch)
            elif ch[0].split('-')[-1] == 'T':
                tails.append(ch)
        #if heads.qsize() == tails.qsize():
        # TODO：当前策略：同等匹配，若头尾数量不符则舍弃多出来的部分
      
        if len(heads) != 0 and len(tails) != 0:
            if len(heads) < len(tails):
                heads += [heads[-1]] * (len(tails) - len(heads))
            if len(heads) > len(tails):
                tails += [tails[-1]] * (len(heads) - len(tails))
    
        for h_t in zip(heads, tails):
            #print(h_t)
            ht = list(h_t) + [cur_relation[i]]
            ht = tuple(ht)
            predict_triples.append(ht)
    return predict_triples


def is_SEO(triples):
    if is_EPO(triples):
        return False
    if is_normal_triple(triples):
        return False
    return True

def is_EPO(triples):
    entity_pairs = set()
    for e in triples:
        e_pair = (e[0], e[1])
        entity_pairs.add(e_pair)
    if len(entity_pairs) != len(triples):
        return True
    return False

def eval(correct_preds, total_preds, total_gt):
    '''
    Evaluation
    :parameter
    :parameter
    :return: P,R,F1
    '''
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_gt if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return p, r, f1


def pretty(l, sen, id2rel):
    # print(l, sen)

    head_words = nltk.word_tokenize(sen + ',')[:-1]
    print("sentence - ", sen)
    print("relation triplet extracted - \n")
    for r in l:
        print((" ".join(head_words[r[0][1]:r[0][2]]), " ".join(head_words[r[1][1]:r[1][2]]), id2rel[r[2]]))


def evaluate(model, loader, label2id, batch_size, rel_num, prefix, id2rel):
    model.eval()
    loader.reset(prefix)
    n = 0
    predictions = []
    final_attn = []
    targets = []
    metrics = {}
    correct_preds = 0.
    total_preds = 0.
    total_gt = 0.
    if prefix == 'dev':
        val_num = loader.dev_len
    else:
        val_num = loader.test_len
    while True:
        with torch.no_grad():
            sents, gts, poses, chars, sen_lens, wrapped, sen = loader.get_batch_dev_test(batch_size, prefix)
            sents = sents
            # print(sen)
            sen_lens = sen_lens
            mask = torch.zeros(sents.size())
            poses = poses
            chars = chars
            n = n + batch_size
            for i in range(sents.size(0)):
                mask[i][:sen_lens[i]] = 1
            sents = sents.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
            poses = poses.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
            chars = chars.repeat([1, rel_num - 1, 1]).view(batch_size * (rel_num - 1), opt.max_len, -1)
            mask = mask.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
            sen_lens = sen_lens.unsqueeze(1).repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1))
            rel = torch.arange(1, rel_num).repeat(batch_size)
            if not opt.use_char:
                chars = None
            if not opt.use_pos:
                poses = None
            # print(sents, sen_lens, rel, mask, poses, chars)
            predict, attention_score = model(sents, sen_lens, rel, mask, poses, chars)   # (batch * rel_num-1) * max_sen_len * label_num
            predict = torch.softmax(predict, -1)
            print(predict.shape, predict[0].shape)
            for i in range(predict.size(0)):
                predict[i][:sen_lens[i], -1] = -1e9
                predict[i][sen_lens[i]:, -1] = 1e9
            decode_tags = np.array(predict.max(-1)[1].data.cpu())
            current_relation = [k for k in range(1, rel_num)]


            for i in range(batch_size):
                triple = tag_mapping(decode_tags[i * (rel_num - 1):(i + 1) * (rel_num - 1)], current_relation, label2id)
                #att = attn_mapping(attention_score[i * (rel_num - 1):(i + 1) * (rel_num - 1)], gts[i])
                target = gts[i]
                predictions.append(triple)
                targets.append(target)

                if n - batch_size + i + 1 <= val_num:
                    
                    print('Sentence %d:' % (n - batch_size + i + 1))
                    # print('predict:')
                    # print(triple)
                    # print('target:')
                    # print(target)
                    pretty(triple, sen[i], id2rel)
                    correct_preds += len(set(triple) & set(target))
                    total_preds += len(set(triple))
                    total_gt += len(set(target))
                    # print(set(triple), set(target))
            if n >= val_num:
                for i in range(n - val_num):
                    predictions.pop()
                    targets.pop()
                p, r, f1, = eval(correct_preds, total_preds, total_gt)
                metrics['P'] = p
                metrics['R'] = r
                metrics['F1'] = f1
                print('test precision {}, recall {}, f1 {}'.format(p * 100, r * 100, f1 * 100))
                break
    model.train()
    return predictions, targets, None, metrics



def multiple_test(predictions, targets):
    preds1, tar1 = [], []
    preds2, tar2 = [], []
    preds3, tar3 = [], []
    preds4, tar4 = [], []
    preds5, tar5 = [], []

    correct_pred1, total_pred1, total_gt1 = 0., 0., 0.
    num1 = 0

    correct_pred2, total_pred2, total_gt2 = 0., 0., 0.
    num2 = 0
    
    correct_pred3, total_pred3, total_gt3 = 0., 0., 0.
    num3 = 0

    correct_pred4, total_pred4, total_gt4 = 0., 0., 0.
    num4 = 0

    correct_pred5, total_pred5, total_gt5 = 0., 0., 0.
    num5 = 0

    for pred, tar in zip(predictions, targets):
        l = len(tar)
        if l == 1:
            preds1.append(pred)
            tar1.append(tar)
            correct_pred1 += len(set(pred) & set(tar))
            total_pred1 += len(set(pred))
            total_gt1 += len(set(tar))
            num1 += 1
        if l == 2:
            preds2.append(pred)
            tar2.append(tar)
            correct_pred2 += len(set(pred) & set(tar))
            total_pred2 += len(set(pred))
            total_gt2 += len(set(tar))
            num2 += 1
        if l == 3:
            preds3.append(pred)
            tar3.append(tar)
            correct_pred3 += len(set(pred) & set(tar))
            total_pred3 += len(set(pred))
            total_gt3 += len(set(tar))
            num3 += 1
        if l == 4:
            preds4.append(pred)
            tar4.append(tar)
            correct_pred4 += len(set(pred) & set(tar))
            total_pred4 += len(set(pred))
            total_gt4 += len(set(tar))
            num4 += 1
        if l >= 5:
            preds5.append(pred)
            tar5.append(tar)    
            correct_pred5 += len(set(pred) & set(tar))
            total_pred5 += len(set(pred))
            total_gt5 += len(set(tar))
            num5 += 1 

    p1, r1, f1 = eval(correct_pred1, total_pred1, total_gt1)
    p2, r2, f2 = eval(correct_pred2, total_pred2, total_gt2)
    p3, r3, f3 = eval(correct_pred3, total_pred3, total_gt3)
    p4, r4, f4 = eval(correct_pred4, total_pred4, total_gt4)
    p5, r5, f5 = eval(correct_pred5, total_pred5, total_gt5)

    print('target relation count 1 metrics: ', 'Precision = ', round(p1, 2), 'Recall = ', round(r1, 2), 'F1 = ', round(f1, 2))
    print('target relation count 3 metrics: ', 'Precision = ', round(p3, 2), 'Recall = ', round(r3, 2), 'F1 = ', round(f3, 2))
    print('target relation count 2 metrics: ', 'Precision = ', round(p2, 2), 'Recall = ', round(r2, 2), 'F1 = ', round(f2, 2))
    print('target relation count 4 metrics: ', 'Precision = ', round(p4, 2), 'Recall = ', round(r4, 2), 'F1 = ', round(f4, 2))
    print('target relation count 5 metrics: ', 'Precision = ', round(p5, 2), 'Recall = ', round(r5, 2), 'F1 = ', round(f5, 2))

def Test(opt):
    loader = Loader(opt)

    if opt.model == 'Rel_based_labeling':
        Model = Rel_based_labeling(opt)
    else:
        raise Exception('Mode name error.')

    if opt.load_test is not None:
        assert os.path.isdir(opt.load_test), "%s must be a path" % opt.load_from
        Model.load_state_dict(torch.load(os.path.join(opt.load_test, 'model-best_cut8.pth')))

    label2id = load_label(opt.input_label2id)
    
    rel2id = json.load(open(opt.input_rel2id, 'r'))
    id2rel = {v:k for k,v in rel2id.items()}

    predictions, targets, attention_score, metrics = evaluate(Model, loader, label2id, opt.eval_batch_size, opt.rel_num, 'test', id2rel)
    
    multiple_test(predictions, targets)

    # print(label2id)
Test(opt)
