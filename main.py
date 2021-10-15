import torch
import random
import numpy as np
import json
from data_processor.processor import AGNewsProcessor, convert_examples_to_features, Collator, CnewsProcessor,\
    FudanNewsProcessor, News20Processor, convert_examples_to_bert_features, BertCollator
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from config import AGNewsConfig, CnewsConfig, FudanNewsConfig, News20Config
from models import CNNClassifier, RNNClassifier, DGCNN, BertClassifier
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report,\
    confusion_matrix
from torch.nn import CrossEntropyLoss
from contrastive_loss import ContrastiveLoss
from vocab import Vocab
from utils.data_process import dingmessage, save_json
import os
import torch.nn.functional as F
from transformers import BertTokenizer


def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(cfg.seed)
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True


def convert_data(examples, label_list, maxlen, vocab2id):
    ids, labels = convert_examples_to_features(examples, label_list, maxlen, vocab2id)
    ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    data = TensorDataset(ids, labels)
    sampler = RandomSampler(data)
    return data, sampler


def convert_bert_data(examples, label_list, maxlen, tokenizer):
    input_ids, mask_ids, label_ids = convert_examples_to_bert_features(examples, label_list, maxlen, tokenizer)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    mask_ids = torch.tensor(mask_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    data = TensorDataset(input_ids, mask_ids, label_ids)
    sampler = RandomSampler(data)
    return data, sampler

def load_data(cfg, device):
    if data_sign == 'agnews':
        data_processor = AGNewsProcessor()
    if data_sign == 'cnews':
        data_processor = CnewsProcessor()
    if data_sign == 'fudan':
        data_processor = FudanNewsProcessor()
    if data_sign == 'news20':
        data_processor = News20Processor()

    label_list = data_processor.get_labels()
    # load data exampels
    if data_mode == 'augmentation':
        train_examples = data_processor.get_train_examples(cfg.train_dir)
        new_train_examples = data_processor.get_train_examples(cfg.new_train_dir)
        train_examples = train_examples + new_train_examples
    elif data_mode == 'no':
        train_examples = data_processor.get_train_examples(cfg.train_dir)
    elif data_mode == 'new':
        train_examples = data_processor.get_train_examples(cfg.new_train_dir)
    dev_examples = data_processor.get_dev_examples(cfg.val_dir)
    if test_mode == 'original':
        test_examples = data_processor.get_test_examples(cfg.test_dir)
    elif test_mode == 'new':
        test_examples = data_processor.get_test_examples(cfg.new_test_dir)
    elif test_mode == 'batch':
        test_examples = data_processor.get_test_examples(cfg.batch_test_dir)

    if replace_mode == 'add_replace':
        replace_tokens = open(cfg.replace_words_dir, 'r', encoding='utf-8').read().splitlines()
    else:
        replace_tokens = ''

    vocab = Vocab(cfg, redo=redo, min_freq=2, max_size=cfg.vocab_size, replace_tokens=replace_tokens)
    vocab2id = vocab.v2i['token']
    if model_type != 'bert':
        collator = Collator(device, vocab2id)
        train_data, train_sampler = convert_data(train_examples, label_list, cfg.maxlen, vocab2id)
        dev_data, dev_sampler = convert_data(dev_examples, label_list, cfg.maxlen, vocab2id)
        test_data, test_sampler = convert_data(test_examples, label_list, cfg.maxlen, vocab2id)
    else:
        collator = BertCollator(device)
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_dir)
        train_data, train_sampler = convert_bert_data(train_examples, label_list, cfg.maxlen, tokenizer)
        dev_data, dev_sampler = convert_bert_data(dev_examples, label_list, cfg.maxlen, tokenizer)
        test_data, test_sampler = convert_bert_data(test_examples, label_list, cfg.maxlen, tokenizer)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, collate_fn=collator,
                                  batch_size=cfg.train_batch_size, num_workers=cfg.nworkers)

    dev_dataloader = DataLoader(dev_data, collate_fn=collator,
                                batch_size=cfg.dev_batch_size, num_workers=cfg.nworkers)

    test_dataloader = DataLoader(test_data, collate_fn=collator,
                                 batch_size=cfg.test_batch_size, num_workers=cfg.nworkers)

    num_train_steps = int(len(train_examples) / cfg.train_batch_size)

    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list, vocab2id


def load_model(cfg, vocab2id, device):

    if model_type == 'cnn':
        model = CNNClassifier(cfg, vocab2id)
    elif model_type == 'rnn':
        model = RNNClassifier(cfg, vocab2id)
    elif model_type == 'dgcnn':
        model = DGCNN(cfg, vocab2id)
    elif model_type == 'bert':
        model = BertClassifier(cfg)
    model.to(device)

    params = model.parameters()
    optimizer = torch.optim.Adam(lr=cfg.learning_rate,
                                params=params)

    return model, optimizer


def eval_checkpoint(model_object, eval_dataloader, cfg, \
                    device, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    eval_accuracy = []
    eval_f1 = []
    eval_recall = []
    eval_precision = []
    logits_all = []
    labels_all = []
    eval_steps = 0

    loss_fct = CrossEntropyLoss()
    cl_loss_fct = ContrastiveLoss(cfg, 0.05)
    for batch in eval_dataloader:
        if model_type != 'bert':
            input_ids, label_ids, pos_in, seq_len = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
        else:
            input_ids, mask_ids, label_ids, pos_in = batch
            input_ids = input_ids.to(device)
            mask_ids = mask_ids.to(device)
            label_ids = label_ids.to(device)

        with torch.no_grad():
            if model_type == 'cnn':
                logits, text_feature = model_object(input_ids)
            elif model_type == 'rnn':
                logits, text_feature = model_object(input_ids, seq_len)
            elif model_type == 'dgcnn':
                logits, text_feature = model_object(input_ids)
            elif model_type == 'bert':
                logits, text_feature = model_object(input_ids, mask_ids)
            tmp_eval_loss = loss_fct(logits, label_ids)
        if model_mode == 'test':
            if save_feature:
                cos_sim = cl_loss_fct.similarity(text_feature, text_feature)
                neg_y = 1 - pos_in.cuda()
                _, hard_mixed = cl_loss_fct.get_hard_cos(text_feature, cos_sim, neg_y)
                hard_mixed = torch.cat(hard_mixed, dim=0)
                text_feature = torch.nn.functional.normalize(text_feature)
                out_feature = torch.cat((text_feature, hard_mixed), dim=0).detach().cpu().numpy().tolist()
                out = []
                label_ids = label_ids.to("cpu").numpy().tolist()
                for i in range(16):
                    out.append({'vector': out_feature[i], 'label_id': label_ids[i], 'label': idx2label[label_ids[i]]})
                for i in range(16, 26):
                    out.append({'vector': out_feature[i], 'label_id': 20, 'label': 'hard'})
                save_json(out, cfg.batch_feature_dir)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)

        eval_loss += tmp_eval_loss.mean().item()

        logits_all.extend(logits.tolist())
        labels_all.extend(label_ids.tolist())
        eval_steps += 1

    average_loss = round(eval_loss / eval_steps, 4)
    eval_accuracy = float(accuracy_score(y_true=labels_all, y_pred=logits_all))
    eval_f1 = float(f1_score(logits_all, labels_all, average='macro'))
    eval_recall = float(recall_score(logits_all, labels_all, average='micro'))
    eval_precision = float(precision_score(logits_all, labels_all, average='micro'))

    if model_mode == 'test':
        # print(classification_report(labels_all, logits_all, target_names=label_list, digits=4))
        confusion = confusion_matrix(labels_all, logits_all)
        list_diag = np.diag(confusion)
        list_raw_sum = np.sum(confusion, axis=1)
        each_acc = list_diag / list_raw_sum
        sample = {'model_name': model_name, 'labels': label_list, 'acc': each_acc.tolist()}
        with open(cfg.each_acc_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return average_loss, eval_accuracy, eval_f1  # eval_precision, eval_recall, eval_f1


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, cfg, \
          device, label_list):
    loss_fct = CrossEntropyLoss()
    cl_loss_fct = ContrastiveLoss(cfg, 0.05)
    max_dev_acc = 0
    max_test_acc = 0
    max_dev_f1 = 0
    max_test_f1 = 0
    total_max_dev_acc = 0
    total_max_test_acc = 0
    total_max_dev_f1 = 0
    total_max_test_f1 = 0
    patience = 0
    dev_acc_list = []
    test_acc_list = []
    for idx in range(int(cfg.epochs)):
        print("#######" * 10)
        print("EPOCH: ", str(idx))
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch_lt_64 = 0
            if model_type != 'bert':
                seq_len = batch[-1]
                batch = tuple(t.to(device) for t in batch[:-1])
                input_ids, label_ids, pos_in = batch
            else:
                batch = tuple(t.to(device) for t in batch)
                input_ids, mask_ids, label_ids, pos_in = batch
            if model_type == 'cnn':
                logits, text_feature = model(input_ids)
            elif model_type == 'rnn':
                logits, text_feature = model(input_ids, seq_len)
            elif model_type == 'dgcnn':
                logits, text_feature = model(input_ids)
            elif model_type == 'bert':
                logits, text_feature = model(input_ids, mask_ids)
            loss = loss_fct(logits, label_ids)
            if cfg.iscl:
                if input_ids.shape[0] != cfg.train_batch_size:
                    total_loss = loss
                    batch_lt_64 += 1
                    assert batch_lt_64 < 2
                else:
                    cl = cl_loss_fct(text_feature, pos_in, hard_mode, cfg.hard_num, device)
                    total_loss = loss + cl*cfg.cl_rate
            else:
                total_loss = loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # if cfg.iscl:
            #     if step % 100 == 0:
            #         print(loss, cl)

        if idx == cfg.change_lr_step:
            for g in optimizer.param_groups:
                g['lr'] = cfg.changed_lr
            print('Learning rate is: %.4f' % (optimizer.param_groups[0]['lr']))

        tmp_dev_loss, tmp_dev_acc, tmp_dev_f1 = eval_checkpoint(model,
                                                                dev_dataloader,
                                                                cfg, device,
                                                                label_list,
                                                                eval_sign="dev")
        dev_acc_list.append((idx+1, tmp_dev_acc))
        if tmp_dev_acc > max_dev_acc:
            max_dev_acc = tmp_dev_acc

        print('dev: ', tmp_dev_loss, tmp_dev_acc, tmp_dev_f1, max_dev_acc)
        tmp_test_loss, tmp_test_acc, tmp_test_f1 = eval_checkpoint(model,
                                                                   test_dataloader,
                                                                   cfg, device,
                                                                   label_list,
                                                                   eval_sign="test")
        test_acc_list.append((idx+1, tmp_test_acc))
        if tmp_test_acc > max_test_acc:
            max_test_acc = tmp_test_acc
        print('test: ', tmp_test_loss, tmp_test_acc, tmp_test_f1, max_test_acc)

        if tmp_dev_acc < max_dev_acc:
            patience += 1
            if patience > cfg.patience:
                break
        else:
            # only save best model
            # torch.save({'state_dict': model.state_dict()}, cfg.checkpoints_dir+'/model_'+str(idx))
            torch.save({'state_dict': model.state_dict()}, cfg.checkpoints_dir+'/best_model')
            patience = 0
            total_max_dev_acc = tmp_dev_acc
            total_max_test_acc = tmp_test_acc
            total_max_dev_f1 = tmp_dev_f1
            total_max_test_f1 = tmp_test_f1
    print('max dev test: %.4f, %.4f' % (total_max_dev_acc, total_max_test_acc))
    dingmessage('%s: cl: %d, augmentation: %d, hard: %s, dev acc/f1: %.4f/%.4f, test acc/f1: %.4f/%.4f' %
                (data_sign, cfg.iscl, int(data_mode=='augmentation'), hard_mode, total_max_dev_acc,
                 total_max_dev_f1, total_max_test_acc, total_max_test_f1))
    acc_dict = {'cl_rate': cfg.cl_rate, 'batch_size': cfg.train_batch_size, 'dev_acc': dev_acc_list,
                'test_acc': test_acc_list}
    # with open(cfg.save_acc_dir, 'a', encoding='utf-8') as f:
    #     f.write(json.dumps(acc_dict, ensure_ascii=False) + '\n')


def test(model, test_dataloader, cfg, device, label_list):
    checkpoint_model = torch.load(cfg.checkpoints_dir+'/best_model')
    model.load_state_dict(checkpoint_model['state_dict'])
    tmp_test_loss, tmp_test_acc, tmp_test_f1 = eval_checkpoint(model,
                                                               test_dataloader,
                                                               cfg, device,
                                                               label_list,
                                                               eval_sign="test")
    print('test: ', tmp_test_loss, tmp_test_acc, tmp_test_f1)
    # dingmessage('test: loss %.4f, acc %.4f, f1 %.4f' % (tmp_test_loss, tmp_test_acc, tmp_test_f1))


def test_wrong_idx(model, eval_dataloader, cfg, \
                    device, label_list):
    checkpoint_model = torch.load(cfg.checkpoints_dir + '/best_model')
    model.load_state_dict(checkpoint_model['state_dict'])
    model.eval()
    logits_all = []
    labels_all = []
    for batch in eval_dataloader:
        input_ids, label_ids, pos_in, seq_len = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits, text_feature = model(input_ids)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)
        logits_all.extend(logits.tolist())
        labels_all.extend(label_ids.tolist())

    eval_accuracy = float(accuracy_score(y_true=labels_all, y_pred=logits_all))
    eval_f1 = float(f1_score(logits_all, labels_all, average='macro'))
    wrong_idx = []
    for idx, i, j in enumerate(zip(labels_all, logits_all)):
        if i != j:
            wrong_idx.append(idx)
    save_json(wrong_idx, cfg.wrong_idx_dir)
    print(eval_accuracy, eval_f1)

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    if data_sign == 'agnews':
        cfg = AGNewsConfig()
    elif data_sign == 'cnews':
        cfg = CnewsConfig()
    elif data_sign == 'fudan':
        cfg = FudanNewsConfig()
    elif data_sign == 'news20':
        cfg = News20Config()
    set_seed(cfg)
    if not os.path.exists(cfg.checkpoints_dir):
        os.makedirs(cfg.checkpoints_dir)
    train_loader, dev_loader, test_loader, num_train_steps, label_list, vocab2id = load_data(cfg, device)
    model, optimizer = load_model(cfg, vocab2id, device)
    if model_mode == 'train':
        train(model, optimizer, train_loader, dev_loader, test_loader, cfg, device, label_list)
    else:
        test(model, test_loader, cfg, device, label_list)
        # test_wrong_idx(model, test_loader, cfg, device, label_list)


if __name__ == '__main__':
    data_sign = 'cnews'
    # data_sign = 'fudan'
    # data_sign = 'agnews'
    # data_sign = 'news20'

    model_name = 'CNN+CL'
    redo = True

    model_type = 'cnn'  # rnn / cnn / dgcnn
    replace_mode = 'no'  # add_replace / no
    test_mode = 'original'  # original / new / batch
    model_mode = 'train'  # train / test
    save_feature = False

    if model_name == 'CNN':
        data_mode, hard_mode= 'no', 'no'
    elif model_name == 'CNN+CL':
        data_mode, hard_mode = 'no', 'normal'
    elif model_name == 'CNN+HM':
        data_mode, hard_mode = 'no', 'single_hard'
    elif model_name == 'CNN+DA':
        data_mode, hard_mode = 'augmentation', 'no'
    elif model_name == 'CNN+CL+HM':
        data_mode, hard_mode = 'no', 'union'
    elif model_name == 'CNN+CL+DA':
        data_mode, hard_mode = 'augmentation', 'normal'
    elif model_name == 'CNN+DA+HM':
        data_mode, hard_mode = 'augmentation', 'single_hard'
    elif model_name == 'CNN+DA+HM+CL':
        data_mode, hard_mode = 'augmentation', 'union'
    main()