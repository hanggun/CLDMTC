#!/usr/bin/env python
# coding:utf-8

import pickle
from collections import Counter
import tqdm
import os
import json
import re

class Vocab(object):
    def __init__(self, config, redo, min_freq=1, special_token=['<PADDING>', '<OOV>'], max_size=None,
                 replace_tokens=None):
        """
        vocabulary class for text classification, initialized from pretrained embedding file
        and update based on minimum frequency and maximum size
        :param config: helper.configure, Configure Object
        :param min_freq: int, the minimum frequency of tokens
        :param special_token: List[Str], e.g. padding and out-of-vocabulary
        :param max_size: int, maximum size of the overall vocabulary
        """
        print('Building Vocabulary....')
        self.corpus_files = {"TRAIN": config.train_dir,
                             "VAL": config.val_dir,
                             "TEST": config.test_dir}
        counter = Counter()
        self.config = config
        # counter for tokens
        self.freqs = {'token': counter.copy(), 'label': counter.copy()}
        # vocab to index
        self.v2i = {'token': dict(), 'label': dict()}
        # index to vocab
        self.i2v = {'token': dict(), 'label': dict()}

        self.min_freq = max(min_freq, 1)
        self.re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9。.,，！!?？]+)")
        token_dir = config.token_dir
        label_dir = config.label_dir
        vocab_dir = {'token': token_dir, 'label': label_dir}
        if os.path.isfile(label_dir) and os.path.isfile(token_dir) and not redo:
            print('Loading Vocabulary from Cached Dictionary...')
            with open(token_dir, 'r', encoding='utf-8') as f_in:
                count = 0
                for i, line in enumerate(f_in):
                    count += 1
                    data = line.rstrip().split('\t')
                    if len(data) != 2:
                        print(count)
                    assert len(data) == 2
                    self.v2i['token'][data[0]] = i
                    self.i2v['token'][i] = data[0]
            with open(label_dir, 'r', encoding='utf-8') as f_in:
                for i, line in enumerate(f_in):
                    data = line.rstrip().split('\t')
                    assert len(data) == 2
                    self.v2i['label'][data[0]] = i
                    self.i2v['label'][i] = data[0]
            for vocab in self.v2i.keys():
                print('Vocabulary of ' + vocab + ' ' + str(len(self.v2i[vocab])))
        else:
            print('Generating Vocabulary from Corpus...')
            self._load_pretrained_embedding_vocab()
            self._count_vocab_from_corpus()
            for vocab in self.freqs.keys():
                print('Vocabulary of ' + vocab + ' ' + str(len(self.freqs[vocab])))

            if replace_tokens:
                self._shrink_vocab('token', config.vocab_size // 2)

                for replace_word in replace_tokens:
                    if len(self.freqs['token']) < config.vocab_size:
                        if replace_word not in self.freqs['token']:
                            self.freqs['token'][replace_word] = self.min_freq
                    else:
                        break
            else:
                self._shrink_vocab('token', max_size)
            for s_token in special_token:
                self.freqs['token'][s_token] = self.min_freq

            for field in self.freqs.keys():
                temp_vocab_list = list(self.freqs[field].keys())
                for i, k in enumerate(temp_vocab_list):
                    self.v2i[field][k] = i
                    self.i2v[field][i] = k
                print('Vocabulary of ' + field + ' with the size of ' + str(len(self.v2i[field].keys())))
                with open(vocab_dir[field], 'w', encoding='utf-8') as f_out:
                    for k in list(self.v2i[field].keys()):
                        f_out.write(k + '\t' + str(self.freqs[field][k]) + '\n')
                print('Save Vocabulary in ' + vocab_dir[field])
        self.padding_index = self.v2i['token']['<PADDING>']
        self.oov_index = self.v2i['token']['<OOV>']

    def _load_pretrained_embedding_vocab(self):
        """
        initialize counter for word in pre-trained word embedding
        """
        pretrained_file_dir = self.config.raw_embedding_dir
        with open(pretrained_file_dir, 'r', encoding='utf8') as f_in:
            print('Loading vocabulary from pretrained embedding...')
            for line in tqdm.tqdm(f_in):
                data = line.rstrip('\n').split(' ')
                if len(data) == 2:
                    # first line in pretrained embedding
                    continue
                v = data[0]
                self.freqs['token'][v] += self.min_freq + 1

    def _count_vocab_from_corpus(self):
        """
        count the frequency of tokens in the specified corpus
        """
        for corpus in self.corpus_files.keys():
            mode = 'ALL'
            with open(self.corpus_files[corpus], 'r', encoding='utf-8') as f_in:
                print('Loading ' + corpus + ' subset...')
                for line in tqdm.tqdm(f_in):
                    data = json.loads(line.rstrip())
                    self._count_vocab_from_sample(data, mode)

    def _count_vocab_from_sample(self, line_dict, mode='ALL'):
        """
        update the frequency from the current sample
        :param line_dict: Dict{'token': List[Str], 'label': List[Str]}
        """
        for k in self.freqs.keys():
            if mode == 'ALL':
                if k == 'token':
                    for t in line_dict[k]:
                        if self.re_han.match(t):
                            self.freqs[k][t] += 1
                else:
                    self.freqs[k][line_dict[k]] += 1
            else:
                for t in line_dict['token']:
                    self.freqs['token'][t] += 1

    def _shrink_vocab(self, k, max_size=None):
        """
        shrink the vocabulary
        :param k: Str, field <- 'token', 'label'
        :param max_size: int, the maximum number of vocabulary
        """
        print('Shrinking Vocabulary...')
        tmp_dict = Counter()
        for v in self.freqs[k].keys():
            if self.freqs[k][v] >= self.min_freq:
                tmp_dict[v] = self.freqs[k][v]
        if max_size is not None:
            tmp_list_dict = tmp_dict.most_common(max_size)
            self.freqs[k] = Counter()
            for (t, v) in tmp_list_dict:
                self.freqs[k][t] = v
        print('Shrinking Vocabulary of tokens: ' + str(len(self.freqs[k])))
