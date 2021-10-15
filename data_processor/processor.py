from utils.data_process import clean_en, save_json, clean_cn, compute_keywords, create_new_data, get_statistics,\
    create_construct_batch, read_json
import pandas as pd
from tqdm import trange, tqdm
import json
from sklearn.model_selection import train_test_split
from config import AGNewsConfig, CnewsConfig, FudanNewsConfig, News20Config
from collections import Counter
import torch
from sklearn.datasets import fetch_20newsgroups
import random
import numpy as np
from collections import defaultdict

random.seed(123)

def convert_examples_to_features(examples, label_list, max_seq_len, vocab2id):
    label_map = {label: i for i, label in enumerate(label_list)}

    input_ids, label_ids = [], []
    for (ex_index, example) in tqdm(enumerate(examples)):
        token = example[0]
        label = example[1]
        input_id = [vocab2id[x] if x in vocab2id else vocab2id['<OOV>'] for x in token]
        if len(input_id) >= max_seq_len:
            input_id = input_id[:max_seq_len]
        else:
            input_id = input_id + [vocab2id['<PADDING>']] * (max_seq_len - len(input_id))
        label_id = label_map[label]
        input_ids.append(input_id)
        label_ids.append(label_id)
    return input_ids, label_ids


def convert_examples_to_bert_features(examples, label_list, max_seq_len, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    input_ids, masks, label_ids = [], [], []
    for (ex_index, example) in tqdm(enumerate(examples)):
        token = example[0][:50]
        token = ''.join(token)
        token = tokenizer.tokenize(token)
        label = example[1]
        if len(token) > max_seq_len - 2:
            token = token[:max_seq_len-2]
        input_x = tokenizer.encode(token)
        input_mask = [1] * len(input_x)
        padding = [0] * (max_seq_len - len(input_x))

        input_x += padding
        input_mask += padding
        label_id = label_map[label]

        input_ids.append(input_x)
        masks.append(input_mask)
        label_ids.append(label_id)

    return input_ids, masks, label_ids


def data_split(data, random_state=42):
    def union(train_x, train_y):
        total = []
        for x,y in zip(train_x, train_y):
            total.append({'token': x, 'label': y})
        return total

    x, y = zip(*[(x['token'], x['label']) for x in data])
    train_x, val_x, train_y, val_y = train_test_split(x,y, test_size=0.1, random_state=random_state)
    train_data = union(train_x, train_y)
    val_data = union(val_x, val_y)
    return train_data, val_data

def read_vocab(vocab_path):
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    vocab2id = {v: i for i, v in enumerate(vocab)}
    id2vocab = {i: v for i, v in enumerate(vocab)}
    return vocab2id, id2vocab


class Collator(object):
    def __init__(self, device, vocab2id):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(Collator, self).__init__()
        self.device = device
        self.vocab2id = vocab2id

    def get_pos(self, targets, device):
        num = len(targets)
        mat = torch.zeros(num, num, dtype=torch.float)
        for i in range(num):
            mat[i] = torch.where(targets == targets[i], torch.tensor(1.0), torch.tensor(0.0))
        return mat

    def __call__(self, batch):
        """
        transform data for training
        :param batch: Dict{'token': List[List[int]],
                           'label': List[List[int]],
                            'token_len': List[int]}
        :return: batch -> Dict{'token': torch.FloatTensor,
                               'label': torch.FloatTensor,
                               'token_len': torch.FloatTensor,
                               'label_list': List[List[int]]}
        """
        label_ids = []
        input_ids = []
        input_length = []
        for sample in batch:
            label_ids.append(sample[1])
            input_ids.append(sample[0].view(1, -1))
            input_length.append(torch.sum(torch.where(sample[0] != self.vocab2id['<PADDING>'],
                                                      torch.tensor(1.0), torch.tensor(0.0))))
        input_ids = torch.cat(input_ids, dim=0)
        label_ids = torch.tensor(label_ids)
        pos_in = self.get_pos(label_ids, self.device)

        return input_ids, label_ids, pos_in, torch.tensor(input_length, dtype=torch.int64)


class BertCollator(object):
    def __init__(self, device):
        """
        Collator object for the collator_fn in data_modules.data_loader
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        """
        super(BertCollator, self).__init__()
        self.device = device

    def get_pos(self, targets, device):
        num = len(targets)
        mat = torch.zeros(num, num, dtype=torch.float)
        for i in range(num):
            mat[i] = torch.where(targets == targets[i], torch.tensor(1.0), torch.tensor(0.0))
        return mat

    def __call__(self, batch):
        """
        transform data for training
        :param batch: Dict{'token': List[List[int]],
                           'label': List[List[int]],
                            'token_len': List[int]}
        :return: batch -> Dict{'token': torch.FloatTensor,
                               'label': torch.FloatTensor,
                               'token_len': torch.FloatTensor,
                               'label_list': List[List[int]]}
        """
        label_ids = []
        mask_ids = []
        input_ids = []
        for sample in batch:
            label_ids.append(sample[2])
            mask_ids.append(sample[1].view(1, -1))
            input_ids.append(sample[0].view(1, -1))
        input_ids = torch.cat(input_ids, dim=0)
        mask_ids = torch.cat(mask_ids, dim=0)
        label_ids = torch.tensor(label_ids)
        pos_in = self.get_pos(label_ids, self.device)

        return input_ids, mask_ids, label_ids, pos_in


class AGNewsProcessor:
    def process_raw_file(self, file_path, subset):
        corpus_data = list()
        df = pd.read_csv(file_path, header=None)
        for i in trange(df.shape[0]):
            label = str(df.iloc[i, 0])
            line = df.iloc[i, 1].rstrip() + df.iloc[i, 2].rstrip()
            sample_tokens = clean_en(line)
            corpus_data.append({'token': sample_tokens, 'label': label})

        return corpus_data

    def preprocess(self, cfg):
        train_corpus_data = self.process_raw_file('../'+cfg.raw_train_dir, 'train')
        test_corpus_data = self.process_raw_file('../'+cfg.raw_test_dir, 'test')
        save_json(train_corpus_data, '../'+cfg.raw_train_json_dir)
        save_json(test_corpus_data, '../'+cfg.test_dir)

    def split_data(self, cfg):
        train_corpus_data = read_json('../'+cfg.raw_train_json_dir)
        train, val = data_split(train_corpus_data, random_state=42)
        save_json(train, '../'+cfg.train_dir)
        save_json(val, '../'+cfg.val_dir)

    def construct_train(self, cfg):
        data = read_json('../'+cfg.train_dir)
        corpus_data = list()
        label_count = defaultdict(int)
        few_labels = self.get_few_labels()
        label_dict = self.get_raw_statistics()
        for item in tqdm(data):
            label = item['label']
            if label in few_labels:
                label_count[label] += 1
                if label_count[label] > label_dict[label] // 5:
                    continue
            corpus_data.append(item)
        save_json(corpus_data, '../'+cfg.construct_train_dir)

    def get_train_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_dev_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_test_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_labels(self):
        return ["1", "2", "3", "4"]
    
    def get_few_labels(self):
        return ['2', '4']

    def get_raw_statistics(self):
        return {'total': 108000, '1': 27029, '3': 27019, '4': 26989, '2': 26963}

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            line = json.loads(line)
            token = line['token']
            label = line['label']
            examples.append((token, label))
        return examples


class CnewsProcessor:
    def process_raw_file(self, file_path, subset):
        corpus_data = list()
        df = open(file_path, 'r', encoding='utf-8')
        for line in tqdm(df):
            line = line.split('\t')
            label = line[0]
            text = line[1]
            sample_tokens = clean_cn(text)
            corpus_data.append({'token': sample_tokens, 'label': label})

        return corpus_data

    def preprocess(self, cfg):
        train_corpus_data = self.process_raw_file('../'+cfg.raw_train_dir, 'train')
        val_corpus_data = self.process_raw_file('../' + cfg.raw_val_dir, 'val')
        test_corpus_data = self.process_raw_file('../'+cfg.raw_test_dir, 'test')
        save_json(train_corpus_data, '../'+cfg.train_dir)
        save_json(val_corpus_data, '../'+cfg.val_dir)
        save_json(test_corpus_data, '../'+cfg.test_dir)

    def construct_train(self, cfg):
        data = read_json('../'+cfg.train_dir)
        corpus_data = list()
        label_count = defaultdict(int)
        few_labels = self.get_few_labels()
        label_dict = self.get_raw_statistics()
        for item in tqdm(data):
            label = item['label']
            if label in few_labels:
                label_count[label] += 1
                if label_count[label] > label_dict[label] // 5:
                    continue
            corpus_data.append(item)
        save_json(corpus_data, '../'+cfg.construct_train_dir)

    def get_train_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_dev_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_test_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_labels(self):
        return ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    def get_few_labels(self):
        return ['财经', '家居', '科技', '时政', '娱乐']

    def get_raw_statistics(self):
        return {'体育': 5000, '娱乐': 5000, '家居': 5000, '房产': 5000, '教育': 5000, '时尚': 5000, '时政': 5000,
                '游戏': 5000, '科技': 5000, '财经': 5000}

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            line = json.loads(line)
            token = line['token']
            label = line['label']
            examples.append((token, label))
        return examples


class FudanNewsProcessor:
    def process_raw_file(self, file_path):
        corpus_data = list()

        df = open(file_path, 'r', encoding='utf-8')
        for line in tqdm(df):
            line = line.split('\t')
            label = line[0]
            text = line[1]
            sample_tokens = clean_cn(text)
            if sample_tokens:
                corpus_data.append({'token': sample_tokens[:400], 'label': label})

        return corpus_data

    def preprocess(self, cfg):
        train_corpus_data = self.process_raw_file('../'+cfg.raw_train_dir)
        test_corpus_data = self.process_raw_file('../'+cfg.raw_test_dir)
        save_json(train_corpus_data, '../'+cfg.raw_train_json_dir)
        save_json(test_corpus_data, '../'+cfg.test_dir)

    def split_data(self, cfg):
        train_corpus_data = read_json('../'+cfg.raw_train_json_dir)
        train, val = data_split(train_corpus_data, random_state=42)
        save_json(train, '../'+cfg.train_dir)
        save_json(val, '../'+cfg.val_dir)

    def get_train_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_dev_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_test_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_labels(self):
        return ['Agriculture', 'Sports', 'Economy', 'Computer', 'Politics', 'Enviornment', 'History', 'Art',
                  'Space', 'Education', 'Energy', 'Law', 'Medical', 'Mine', 'Electronics', 'Transport', 'Philosophy',
                  'Military', 'Literature', 'Communication']
    
    def get_few_labels(self):
        return ['Education', 'Energy', 'Law', 'Medical', 'Mine', 'Electronics',
                       'Transport', 'Philosophy', 'Military', 'Literature', 'Communication']

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            line = json.loads(line)
            token = line['token']
            label = line['label']
            examples.append((token, label))
        return examples


class News20Processor:
    def process_raw_file(self, file_path, subset):
        corpus_data = list()
        data = fetch_20newsgroups(data_home=file_path, subset=subset)
        for i in trange(len(data.data)):
            label = data.target_names[data.target[i]]
            line = data['data'][i]
            sample_tokens = clean_en(line)
            corpus_data.append({'token': sample_tokens, 'label': label})

        return corpus_data

    def preprocess(self, cfg):
        train_corpus_data = self.process_raw_file('../'+cfg.data_dir, 'train')
        test_corpus_data = self.process_raw_file('../'+cfg.data_dir, 'test')
        save_json(train_corpus_data, '../'+cfg.raw_train_json_dir)
        save_json(test_corpus_data, '../'+cfg.test_dir)

    def split_data(self, cfg):
        train_corpus_data = read_json('../'+cfg.raw_train_json_dir)
        train, val = data_split(train_corpus_data, random_state=42)
        save_json(train, '../'+cfg.train_dir)
        save_json(val, '../'+cfg.val_dir)

    def construct_train(self, cfg):
        data = read_json('../'+cfg.train_dir)
        corpus_data = list()
        label_count = defaultdict(int)
        few_labels = self.get_few_labels()
        label_dict = self.get_raw_statistics()
        for item in tqdm(data):
            label = item['label']
            if label in few_labels:
                label_count[label] += 1
                if label_count[label] > label_dict[label] // 5:
                    continue
            corpus_data.append(item)
        save_json(corpus_data, '../'+cfg.construct_train_dir)

    def get_tf_idf(self, cfg):
        compute_keywords(cfg)

    def get_train_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_dev_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_test_examples(self, data_path):
        return self._create_examples(
            open(data_path, 'r', encoding='utf-8'))

    def get_labels(self):
        return ['soc.religion.christian', 'comp.os.ms-windows.misc', 'sci.med', 'rec.autos',
                'sci.crypt', 'sci.electronics', 'comp.sys.ibm.pc.hardware', 'rec.motorcycles',
                'sci.space', 'rec.sport.baseball', 'comp.graphics', 'rec.sport.hockey',
                'misc.forsale', 'comp.windows.x', 'talk.politics.mideast',
                'comp.sys.mac.hardware', 'talk.politics.guns', 'talk.politics.misc',
                'alt.atheism', 'talk.religion.misc']

    def get_few_labels(self):
        return ['soc.religion.christian', 'sci.med', 'sci.crypt', 'comp.sys.ibm.pc.hardware', 'sci.space',
                'comp.graphics', 'misc.forsale', 'talk.politics.mideast', 'talk.politics.guns',
                'alt.atheism']

    def get_raw_statistics(self):
        return {'total': 10182, 'soc.religion.christian': 547, 'rec.sport.hockey': 547, 'rec.motorcycles': 546,
                'sci.crypt': 543, 'rec.sport.baseball': 542, 'comp.graphics': 538, 'comp.windows.x': 538,
                'comp.os.ms-windows.misc': 537, 'sci.med': 535, 'rec.autos': 530, 'comp.sys.ibm.pc.hardware': 528,
                'sci.electronics': 526, 'misc.forsale': 525, 'sci.space': 525, 'comp.sys.mac.hardware': 511,
                'talk.politics.mideast': 505, 'talk.politics.guns': 485, 'alt.atheism': 433,
                'talk.politics.misc': 402, 'talk.religion.misc': 339}

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            line = json.loads(line)
            token = line['token']
            label = line['label']
            examples.append((token, label))
        return examples

def compute_rank_value(sample, ctfidf):
    """计算文本中与label最相关的词，排序输出"""
    tokens, label = sample
    res = dict()
    new_tokens = []
    for token in tokens:
        if token not in new_tokens:
            new_tokens.append(token)
    tokens = new_tokens
    for w in tokens:
        if w in ctfidf:
            idf = ctfidf[w]['idf']
            label_dict = ctfidf[w]['label']
            tmp_dict = label_dict.copy()

            if label in label_dict:
                wc = label_dict[label]
                del tmp_dict[label]
            else:
                wc = 0
            if tmp_dict:
                wm = np.mean(list(tmp_dict.values()))
                wv = np.var(list(tmp_dict.values()))
                if wv == 0:
                    wv = 1.0
            else:
                wm, wv = 0, 1.0
            r = (wc - wm) / wv
            v = r * idf
            res[w] = v
    res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    return res


def save_rank_value(cfg, input_dir, output_dir, ctfidf):
    datasets = []
    with open('../' + input_dir, 'r', encoding='utf-8') as f:
        for idx1, line in enumerate(f):
            line = json.loads(line)
            tokens = line['token'][:cfg.maxlen]
            label = line['label']
            datasets.append((tokens, label))

    rank_values = []
    for data in tqdm(datasets):
        rank_values.append(compute_rank_value(data, ctfidf))
    with open('../' + output_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(rank_values, ensure_ascii=False, indent=2))
    return rank_values


def adjust_data(file):
    from collections import defaultdict
    data = read_json(file)
    data_dict = defaultdict(list)
    out = []
    for line in data:
        data_dict[line['label']].append(line)

if __name__ == '__main__':
    agnews_cfg = AGNewsConfig()
    cnews_cfg = CnewsConfig()
    fudan_news_cfg = FudanNewsConfig()
    news20_cfg = News20Config()
    ag = AGNewsProcessor()
    cn = CnewsProcessor()
    fd = FudanNewsProcessor()
    n2 = News20Processor()
    # ag.preprocess(agnews_cfg)
    # ag.split_data(agnews_cfg)
    # ag.construct_train(agnews_cfg)
    # create_new_data(agnews_cfg, agnews_cfg.train_dir, agnews_cfg.new_train_dir, agnews_cfg.rank_value_dir,
    #                                  0.02, ag.get_labels(), mode='useful')
    # get_statistics(agnews_cfg.train_dir)
    # get_statistics(agnews_cfg.construct_train_dir)
    # get_statistics(agnews_cfg.val_dir)
    # get_statistics(agnews_cfg.test_dir)


    # compute_keywords(agnews_cfg)
    # cn.preprocess(cnews_cfg, is_construct=False)
    # create_new_data(cnews_cfg, cnews_cfg.train_dir, cnews_cfg.new_train_dir, cnews_cfg.rank_value_dir,
    #                                  0.02, cn.get_labels(), mode='useful')
    # get_statistics(cnews_cfg.train_dir)
    # get_statistics(cnews_cfg.val_dir)
    # get_statistics(cnews_cfg.test_dir)


    # fd.preprocess(fudan_news_cfg)
    # fd.split_data(fudan_news_cfg)
    # create_new_data(fudan_news_cfg, fudan_news_cfg.train_dir, fudan_news_cfg.new_train_dir,
    #                 fudan_news_cfg.rank_value_dir, 0.02, fd.get_few_labels(), mode='useful')
    # with open('../'+fudan_news_cfg.rank_value_dir, 'r', encoding='utf-8') as f:
    #     a = json.load(f)
    # with open('../'+fudan_news_cfg.rank_value_dir1, 'r', encoding='utf-8') as f:
    #     b = json.load(f)
    # print(a==b)
    # get_statistics(fudan_news_cfg.train_dir)
    # get_statistics(fudan_news_cfg.val_dir)
    # get_statistics(fudan_news_cfg.test_dir)
    # compute_keywords(fudan_news_cfg)


    # n2.preprocess(news20_cfg)
    # n2.split_data(news20_cfg)
    # n2.construct_train(news20_cfg)
    # n2.get_tf_idf(news20_cfg)
    # create_new_data(news20_cfg, news20_cfg.train_dir, news20_cfg.new_train_dir,
    #                 news20_cfg.rank_value_dir, 0.02, n2.get_labels(), mode='useful')
    # get_statistics(news20_cfg.train_dir)
    # get_statistics(news20_cfg.construct_train_dir)
    # get_statistics(news20_cfg.val_dir)
    # get_statistics(news20_cfg.test_dir)
    # create_construct_batch(news20_cfg)
    # get_statistics(news20_cfg.batch_test_dir)

