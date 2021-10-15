import re
import json
import jieba
import requests
from datetime import datetime
from tqdm import tqdm
import numpy as np
from config import AGNewsConfig, CnewsConfig, FudanNewsConfig, News20Config
import random
from collections import Counter
# from data_processor.multiprocessor import compute_rank_value
random.seed(1234)
np.random.seed(1234)


english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                     'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                     "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                     "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                     'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                     'won', "won't", 'wouldn', "wouldn't", '\\.', '\\?', ',', '\\!', "'s", '', '\\(', '\\)', '!']


def build_vocab(cfg):
    from collections import Counter
    files = [cfg.train_dir, cfg.val_dir, cfg.test_dir]
    counter = Counter()
    for file in files:
        with open('../' + file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                token = line['token']
                for word in token:
                    counter[word] += 1
    words = []
    with open('../' + cfg.vocab_dir, 'w', encoding='utf-8') as f:
        for idx, (word, value) in tqdm(enumerate(counter.most_common())):
            words.append(word + '\n')
            if idx + 1 == cfg.vocab_size - 2:
                break
        f.write(''.join(words))
        f.write('<PADDING>' + '\n')
        f.write('<OOV>' + '\n')


class CleanEnglsih:
    def clean_stopwords(self, sample):
        """
        :param sample: List[Str], lower case
        :return:  List[Str]
        """
        return [token for token in sample if token not in english_stopwords]

    def clean_str(self, string):
        """
        Original Source:  https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        :param string: Str
        :return -> Str
        """
        string = string.strip().strip('"')
        string = re.sub(r"[^A-Za-z(),!?\.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"\.", " \. ", string)
        string = re.sub(r"\"", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def preprocess_line(self, sample):
        """
        :param sample: Str, "The sample would be tokenized and filtered according to the stopwords list"
        :return: token_list -> List[Str]
        """
        sample = self.clean_str(sample.lstrip().rstrip()).split()
        token_list = self.clean_stopwords(sample)
        return token_list


class CleanChinese:
    def clean_char(self, sample):
        chars = []
        re_han = re.compile('[\u4e00-\u9fa5a-zA-Z]')
        for word in sample:
            if re_han.match(word):
                chars.append(word)
        return chars

    def clean_words(self, sample):
        re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")
        blocks = re_han.split(sample.strip())
        tokens = []
        for blk in blocks:
            if re_han.match(blk):
                for w in jieba.cut(blk):
                    tokens.append(w)

        return tokens

    def preprocess_line(self, sample):
        """
        :param sample: Str, "The sample would be tokenized and filtered according to the stopwords list"
        :return: token_list -> List[Str]
        """
        token_list = self.clean_words(sample)
        return token_list


def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def read_json(file):
    datasets = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            datasets.append(json.loads(line))
    return datasets

def dingmessage(msg=''):
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=133652f86733520d80b513ec9e06b25b17b023b848cd9c1825e2e8c00ef843ea"
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
}
    tex = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' 程序完成啦 ' + msg
    message ={
        "msgtype": "text",
        "text": {
            "content": tex
        },
        "at": {
            "isAtAll": True
        }
    }
    message_json = json.dumps(message)
    requests.post(url=webhook, data=message_json, headers=header)


def compute_keywords(config):
    res = dict()
    N = 0
    for filename in [config.train_dir, config.val_dir, config.test_dir]:
        with open('../'+filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)
                content = line['token']
                label = line['label']
                for w in set(content):
                    c = content.count(w)
                    if w in res:
                        res[w]['count'] += 1
                        if label in res[w]['label']:
                            res[w]['label'][label] += c
                        else:
                            res[w]['label'] = dict(res[w]['label'], **{label: c})
                    else:
                        res[w] = {}
                        res[w]['count'] = 1
                        res[w]['label'] = {label: c}

                N += 1

    dataset = dict()

    for w in res:
        dataset[w] = {}
        dataset[w]['label'] = res[w]['label']
        dataset[w]['idf'] = np.math.log(N / res[w]['count'])

    with open('../'+config.tfidf_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, ensure_ascii=False, indent=2))


def replace_sim_word(word, words_similar, new_content, replace_words, replace_words_dict):
    replace_flag = False
    if word in words_similar:
        sim_word = words_similar[word][0][0]
        sim_value = words_similar[word][0][1]
        if sim_value > 0.6:
            idx = new_content.index(word)
            new_content[idx] = sim_word
            if word + '\n' not in replace_words_dict:
                replace_words.append(word + '\n')
                replace_words_dict[word + '\n'] = 1
            if sim_word + '\n' not in replace_words_dict:
                replace_words.append(sim_word + '\n')
                replace_words_dict[sim_word + '\n'] = 1
            replace_flag = True
    return replace_flag, new_content, replace_words, replace_words_dict


def get_topk_ranks(new_content, rank_value, max_replace_num):
    topk_rank_value_exist_in_content = []
    rank_word_count = 0
    for word, value in rank_value:
        assert word in new_content
        topk_rank_value_exist_in_content.append(word)
        rank_word_count += 1
        if rank_word_count == max_replace_num*2+10:
            break
    return topk_rank_value_exist_in_content


def create_new_data(cfg, data_dir, output_dir, rank_value_dir, replace_percent, few_labels, mode='useful'):
    with open('../'+cfg.words_similar_dir, 'r', encoding='utf-8') as f:
        words_similar = json.load(f)
    with open('../'+rank_value_dir, 'r', encoding='utf-8') as f:
        rank_values = json.load(f)
    words_all = open('../'+cfg.words_all, 'r', encoding='utf-8').read().splitlines()
    replace_words = []
    replace_words_dict = {}
    corpus_data = list()
    datasets = []
    with open('../'+data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            datasets.append(line)
    # random_order = np.random.permutation(range(len(datasets)))
    # datasets = [datasets[x] for x in random_order]
    # rank_values = [rank_values[x] for x in random_order]
    for idx1, line in tqdm(enumerate(datasets)):
        line = json.loads(line)
        tokens = line['token']
        label = line['label']
        new_content = tokens[:cfg.maxlen].copy()  # to make sure only the word exists in top maxlen will be replaced
        max_replace_num = int(len(new_content) * replace_percent)
        num = 0
        random_replace_count = 0
        if mode == 'useful':
            topk_rank_value_exist_in_content = get_topk_ranks(new_content, rank_values[idx1], max_replace_num)
        elif mode == 'useless':
            topk_rank_value_exist_in_content = get_topk_ranks(new_content, rank_values[idx1][::-1], max_replace_num)
        elif mode == 'keyword_random_replace':
            topk_rank_value_exist_in_content = get_topk_ranks(new_content, rank_values[idx1], max_replace_num)
            if label in few_labels:
                for word in topk_rank_value_exist_in_content:
                    idx = new_content.index(word)
                    random_word = random.choice(words_all)
                    new_content[idx] = random_word
                    random_replace_count += 1
                    if random_replace_count == max_replace_num:
                        break
        elif mode == 'random_replace':
            if label in few_labels:
                for _ in range(max_replace_num):
                    random_idx = random.sample(range(len(new_content)), 1)[0]
                    random_word = random.choice(words_all)
                    new_content[random_idx] = random_word
        elif mode == 'useful_deletion':
            topk_rank_value_exist_in_content = get_topk_ranks(new_content, rank_values[idx1], max_replace_num)
            if label in few_labels:
                for word in topk_rank_value_exist_in_content:
                    idx = new_content.index(word)
                    new_content.pop(idx)
                    random_replace_count += 1
                    if random_replace_count == max_replace_num:
                        break
                # corpus_data.append({'token': new_content, 'label': label})
        elif mode == 'random_deletion':
            if label in few_labels:
                for _ in range(max_replace_num):
                    random_idx = random.sample(range(len(new_content)), 1)[0]
                    new_content.pop(random_idx)
                # corpus_data.append({'token': new_content, 'label': label})
        elif mode == 'crop':
            if label in few_labels:
                random_idx = random.sample(range(len(new_content)-max_replace_num), 1)[0]
                for _ in range(max_replace_num):
                    new_content.pop(random_idx)
                # corpus_data.append({'token': new_content, 'label': label})
        else:
            raise RuntimeError('error')
        if mode == 'useful' or mode == 'useless':
            if label in few_labels:
                for word in topk_rank_value_exist_in_content:  # ensure only top k ranked value will be replaced
                    flag, new_content, replace_words, replace_words_dict = replace_sim_word(word,
                                                                                            words_similar,
                                                                                            new_content,
                                                                                            replace_words,
                                                                                            replace_words_dict)
                    if flag:
                        num += 1
                    if num == max_replace_num:
                        break
                # corpus_data.append({'token': new_content, 'label': label})
        if label in few_labels:
            # corpus_data.append({'token': tokens, 'label': label})
            corpus_data.append({'token': new_content, 'label': label})

    with open('../'+cfg.replace_words_dir, 'w', encoding='utf-8') as f:
        f.write(''.join(replace_words))

    save_json(corpus_data, '../'+output_dir)


def get_statistics(file_path):
    counter = Counter()
    total = 0
    avg_length = 0
    datasets = []
    with open('../'+file_path, 'r', encoding='utf-8') as f:
        for line in f:
            datasets.append(line)
            line = json.loads(line)
            tokens = line['token']
            avg_length += len(tokens)
            label = line['label']
            total += 1
            counter[label] += 1
            counter['total'] += 1
    print(counter)
    print('average length %.4f' % (avg_length / total))


def create_construct_batch(cfg):
    corpus_data = []
    datasets = []
    count = 0
    with open('../'+cfg.train_dir, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            datasets.append(line)
    compared_label = datasets[0]['label']
    compared_text = datasets[0]['token']
    corpus_data.append(datasets[0])
    random.shuffle(datasets)
    for line in datasets:
        if line['label'] != compared_label:
            corpus_data.append(line)
            count += 1
            if count == 14:
                break
    with open('../'+cfg.words_similar_dir, 'r', encoding='utf-8') as f:
        words_similar = json.load(f)
    new_content = compared_text.copy()
    rank_value = compute_rank_value((compared_text, compared_label))
    max_replace_num = int(len(new_content)*0.02)
    topk_rank_value_exist_in_content = get_topk_ranks(new_content, rank_value, max_replace_num)
    num = 0
    for word in topk_rank_value_exist_in_content:
        flag, new_content, replace_words, replace_words_dict = replace_sim_word(word,
                                                                                words_similar,
                                                                                new_content,
                                                                                [],
                                                                                {})
        if flag:
            num += 1
        if num == max_replace_num:
            break
    corpus_data.append({'token': new_content, 'label': compared_label})
    save_json(corpus_data, '../'+cfg.batch_test_dir)


clean_en = CleanEnglsih().preprocess_line
clean_cn = CleanChinese().preprocess_line


if __name__ == '__main__':
    dingmessage()