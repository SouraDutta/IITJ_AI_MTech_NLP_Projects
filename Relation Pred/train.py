import torch
import numpy as np
from DataLoader import Loader
import os
import config
import torch
import torch.nn as nn
import torch.optim as optim
import six
from six.moves import cPickle
import model
import traceback
import time
import json
from Rel_based_labeling import Rel_based_labeling


opt = config.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

def load_label(input_label2id):
    label2id = json.load(open(input_label2id, 'r'))
    return label2id

def pickle_load(f):
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)

def pickle_dump(obj, f):
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def build_optimizer(params, opt):
    if opt.optimizer == 'adam':
        return optim.Adam(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay, nesterov=True)
    else:
        raise Exception('optimizer is invalid.')

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, mask):
        loss_total = -input.gather(dim=2, index=target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss_total)
        return loss

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


def save_checkpoint(model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
    torch.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos%s.pkl' %(append)), 'wb') as f:
        pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories%s.pkl' %(append)), 'wb') as f:
            pickle_dump(histories, f)

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

def evaluate(model, loader, label2id, batch_size, rel_num, prefix):
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
            print(sents, sen_lens, rel, mask, poses, chars)
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
                    print('predict:')
                    print(triple)
                    print('target:')
                    print(target)
                    
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

class LossWrapper(nn.Module):
    def __init__(self, Model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = Model
        self.criterion = CrossEntropy()

    def forward(self, sent, sen_len, rel, mask, label, mask2, poses=None, chars=None):
        # print(rel)
        predict, weight = self.model(sent, sen_len, rel, mask, poses, chars)
        predict = torch.log_softmax(predict, dim=-1)
        # print(label)
        loss = self.criterion(predict, label, mask2)
        return loss


def train(opt):
    loader = Loader(opt)
    infos = {}
    histories = {}

    if opt.model == 'Rel_based_labeling':
        Model = Rel_based_labeling(opt)
        print(Model)
    else:
        raise Exception('Mode name error.')

    if opt.load_from is not None:
        assert os.path.isdir(opt.load_from), "%s must be a path" % opt.load_from
        Model.load_state_dict(torch.load(os.path.join(opt.load_from, 'model-best.pth')))

    LW_model = LossWrapper(Model, opt)
    # DP_lw_model = torch.nn.DataParallel(LW_model)
    LW_model.train()
    optimizer = build_optimizer(Model.parameters(), opt)

    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos-best.pkl'), 'rb') as f:
            infos = pickle_load(f)

        if os.path.isfile(os.path.join(opt.start_from, 'histories-best.pkl')):
            with open(os.path.join(opt.start_from, 'histories-best.pkl'), 'rb') as f:
                histories = pickle_load(f)

        if os.path.isfile(os.path.join(opt.start_from, 'optimizer-best.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-best.pth')))
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['opt'] = opt
        infos['label2id'] = load_label(opt.input_label2id)

    iteration = infos.get('iter', '0')
    epoch = infos.get('epoch', '0')
    best_val_score = infos.get('best_val_score', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    epoch_done = True
    best_epoch = -1
    try:
        while True:
            if epoch_done:
                iteration = 0
                if epoch != 0:
                    predictions, targets, _ ,metrics = evaluate(Model, loader, infos['label2id'], opt.eval_batch_size, opt.rel_num, 'test')
                    val_result_history[iteration] = {'predictions': predictions, 'metrics': metrics, 'targets': targets}
                    #print('dev res: ', metrics)
                    current_score = metrics['F1']
                    histories['c'] = val_result_history
                    histories['loss_history'] = loss_history
                    histories['lr_history'] = lr_history

                    best_flag = False
                    print(metrics)
                    if current_score > best_val_score:
                        best_epoch = epoch
                        best_val_score = current_score
                        best_flag = True
                    infos['best_val_score'] = best_val_score

                    save_checkpoint(Model, infos, optimizer, histories)
                    if best_flag:
                        save_checkpoint(Model, infos, optimizer, append='best')


                epoch_done = False
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                set_lr(optimizer, opt.current_lr)
            start = time.time()
            data = loader.get_batch_train(opt.batch_size)
            #data = sorted(data, key=lambda x: x[-1], reverse=True)
            wrapped = data[-1]
            data = data[:-1]
            #print('Read data:', time.time() - start)

            # torch.cuda.synchronize()
            start = time.time()
            data = [t for t in data]
            # print(data)
            sents, rels, labels, poses, chars, sen_lens = data
            # print(rels)
            if not opt.use_char:
                chars = None
            if not opt.use_pos:
                poses = None
            mask = torch.zeros(sents.size())
            for i in range(sents.size(0)):
                mask[i][:sen_lens[i]] = 1

            mask2 = torch.where(labels == 8, torch.ones_like(sents), torch.ones_like(sents)*10)
            mask2 = mask2.float() * mask.float()

            optimizer.zero_grad()
            # print(rels, labels)
            sum_loss = LW_model(sents, sen_lens, rels, mask, labels, mask2, poses, chars)

            loss = sum_loss/sents.shape[0]
            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.item()
            # torch.cuda.synchronize()
            if iteration % 200 == 0:
                end = time.time()
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))

            iteration += 1
            if wrapped:
                epoch += 1
                epoch_done = True
            infos['iter'] = iteration
            infos['epoch'] = epoch

            if iteration % opt.save_loss_every == 0:
                loss_history[iteration] = train_loss
                lr_history[iteration] = opt.current_lr
            if opt.max_epoch != -1 and epoch >= opt.max_epoch:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        save_checkpoint(Model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

if __name__ == '__main__':
    train(opt)
