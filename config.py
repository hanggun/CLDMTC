# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 22:33:37 2021

@author: Ps
"""

class AGNewsConfig:
    def __init__(self):
        self.embedding_size = 300
        self.maxlen = 60
        self.num_classes = 4
        self.vocab_size = 40000
        self.keep_prob = 0.5
        self.train_batch_size = 64
        self.dev_batch_size = 256
        self.test_batch_size = 256
        self.nworkers = 0
        self.epochs = 100
        self.num_filters = 128
        self.filter_sizes = [2, 3, 4]
        self.learning_rate = 1e-3
        self.seed = 1234
        self.data_sign = 'agnews'
        self.patience = 8
        self.change_lr_step = 2
        self.changed_lr = 1e-3
        self.iscl = 0
        self.cl_rate = 0
        self.rnn_hidden_size = 64
        self.use_layer_norm = True
        self.use_pretrain = True

        self.rank_value_dir = 'data/AGNews/rank_value.json'
        self.check_checkpoints_dir = 'checkpoints/AGNews/check'
        self.tfidf_dir = 'data/AGNews/ctfidf.json'
        self.words_similar_dir = 'data/AGNews/words_similar.json'
        self.checkpoints_dir = 'checkpoints/AGNews'
        self.raw_train_dir = 'data/AGNews/train.csv'
        self.raw_test_dir = 'data/AGNews/test.csv'
        self.train_dir = 'data/AGNews/train.json'
        self.val_dir = 'data/AGNews/val.json'
        self.test_dir = 'data/AGNews/test.json'
        self.raw_train_json_dir = 'data/AGNews/raw_train.json'
        self.construct_train_dir = 'data/AGNews/construct_train.json'
        self.new_train_dir = 'data/AGNews/new_train.json'
        self.data_dir = 'data/AGNews/data.pkl'
        self.log_level = 'info'
        self.log_filename = 'logs/log.log'
        self.token_dir = 'data/AGNews/token.txt'
        self.label_dir = 'data/AGNews/label.txt'
        self.raw_vocab_dir = 'data/AGNews/raw_vocab.txt'
        self.vocab_dir = 'data/AGNews/vocab.txt'
        self.raw_embedding_dir = 'pretrained_models/glove.6B.300d.txt'
        self.embedding_dir = 'pretrained_models/GoogleNews-vectors-negative300_filtered.txt'
        self.check_dir = 'data/AGNews/check.txt'
        self.replace_words_dir = 'data/AGNews/replace_words.txt'
        self.words_all = 'data/AGNews/words_all.txt'
        self.save_acc_dir = 'data/AGNews/save_acc.json'
        self.each_acc_dir = 'data/AGNews/each_acc.json'
        self.bert_dir = 'roberta'


class CnewsConfig:
    def __init__(self):
        self.embedding_size = 300
        self.maxlen = 300
        self.num_classes = 10
        self.vocab_size = 40000
        self.keep_prob = 0.5 # 0.2 for CNN and CNN + CL/ 0.5 for DA
        self.train_batch_size = 64
        self.dev_batch_size = 256
        self.test_batch_size = 256
        self.epochs= 100 # epoch 1 for hm+da and hm+da+cl
        self.num_filters = 128
        self.filter_sizes = [2, 3, 4]
        self.nworkers = 0
        self.learning_rate = 1e-3
        self.seed = 1234
        self.data_sign = 'cnews'
        self.patience = 8
        self.change_lr_step = 2
        self.changed_lr = 1e-3
        self.iscl = 1
        self.cl_rate = 0.1
        self.rnn_hidden_size = 64
        self.hard_num = 10
        self.use_layer_norm = True
        self.use_pretrain = True

        self.rank_value_dir = 'data/cnews/rank_value.json'
        self.test_rank_value_dir = 'data/cnews/test_rank_value.json'
        self.check_checkpoints_dir = 'checkpoints/cnews/check'
        self.tfidf_dir = 'data/cnews/ctfidf.json'
        self.words_similar_dir = 'data/cnews/words_similar.json'
        self.checkpoints_dir = 'checkpoints/cnews'
        self.raw_train_dir = 'data/cnews/cnews.train.txt'
        self.raw_test_dir = 'data/cnews/cnews.test.txt'
        self.raw_val_dir = 'data/cnews/cnews.val.txt'
        self.train_dir = 'data/cnews/train.json'
        self.raw_train_json_dir = 'data/cnews/raw_train.json'
        self.construct_train_dir = 'data/cnews/construct_train.json'
        self.new_train_dir = 'data/cnews/new_train.json'
        self.new_test_dir = 'data/cnews/new_test.json'
        self.val_dir = 'data/cnews/val.json'
        self.test_dir = 'data/cnews/test.json'
        self.data_dir = 'data/cnews/data.pkl'
        self.log_level = 'info'
        self.log_filename = 'logs/log.log'
        self.token_dir = 'data/cnews/token.txt'
        self.label_dir = 'data/cnews/label.txt'
        self.raw_vocab_dir = 'data/cnews/raw_vocab.txt'
        self.vocab_dir = 'data/cnews/vocab.txt'
        self.raw_embedding_dir = 'pretrained_models/sogou.txt'
        self.embedding_dir = 'pretrain_models/GoogleNews-vectors-negative300_filtered.txt'
        self.check_dir = 'data/cnews/check.txt'
        self.words_all = 'data/cnews/words_all.txt'
        self.replace_words_dir = 'data/cnews/replace_words.txt'
        self.save_acc_dir = 'data/cnews/save_acc_iter1.json'
        self.each_acc_dir = 'data/cnews/each_acc.json'
        self.bert_dir = 'roberta'


class News20Config:
    def __init__(self):
        self.embedding_size = 300
        self.maxlen = 300
        self.num_classes = 20
        self.vocab_size = 40000
        self.keep_prob = 0.5
        self.train_batch_size = 64
        self.dev_batch_size = 256
        self.test_batch_size = 256
        self.nworkers = 0
        self.epochs = 100
        self.num_filters = 128
        self.filter_sizes = [2, 3, 4]
        self.learning_rate = 1e-3
        self.seed = 1234
        self.data_sign = 'news20'
        self.patience = 8
        self.change_lr_step = 4
        self.changed_lr = 1e-3
        self.iscl = 0
        self.cl_rate = 0
        self.rnn_hidden_size = 64
        self.use_layer_norm = True
        self.use_pretrain = True

        self.rank_value_dir = 'data/news20/rank_value.json'
        self.test_rank_value_dir = 'data/news20/test_rank_value.json'
        self.check_checkpoints_dir = 'checkpoints/news20/check'
        self.tfidf_dir = 'data/news20/ctfidf.json'
        self.words_similar_dir = 'data/news20/words_similar.json'
        self.checkpoints_dir = 'checkpoints/news20'
        self.train_dir = 'data/news20/train.json'
        self.raw_train_json_dir = 'data/news20/raw_train.json'
        self.construct_train_dir = 'data/news20/construct_train.json'
        self.new_train_dir = 'data/news20/new_train.json'
        self.new_test_dir = 'data/news20/new_test.json'
        self.batch_test_dir = 'data/news20/batch_test.json'
        self.val_dir = 'data/news20/val.json'
        self.test_dir = 'data/news20/test.json'
        self.data_dir = 'data/news20/'
        self.log_level = 'info'
        self.log_filename = 'logs/log.log'
        self.token_dir = 'data/news20/token.txt'
        self.label_dir = 'data/news20/label.txt'
        self.raw_vocab_dir = 'data/news20/raw_vocab.txt'
        self.vocab_dir = 'data/news20/vocab.txt'
        self.raw_embedding_dir = 'pretrained_models/glove.6B.300d.txt'
        self.embedding_model_dir = 'pretrained_models/glove.6B.300d.txt'
        self.embedding_dir = 'pretrained_models/glove.6B.300d.txt'
        self.check_dir = 'data/news20/check.txt'
        self.words_all = 'data/news20/words_all.txt'
        self.replace_words_dir = 'data/news20/replace_words.txt'
        self.batch_feature_dir = 'data/news20/batch_feature.json'
        self.save_acc_dir = 'data/news20/save_acc.json'
        self.each_acc_dir = 'data/news20/each_acc.json'
        self.bert_dir = 'roberta'

class FudanNewsConfig:
    def __init__(self):
        self.embedding_size = 300
        self.maxlen = 300
        self.num_classes = 20
        self.vocab_size = 40000
        self.keep_prob = 0.5
        self.train_batch_size = 64
        self.dev_batch_size = 256
        self.test_batch_size = 256
        self.epochs = 100
        self.num_filters = 128
        self.filter_sizes = [2, 3, 4]
        self.nworkers = 0
        self.learning_rate = 1e-3
        self.seed = 1234
        self.data_sign = 'fudan'
        self.patience = 3
        self.change_lr_step = 10
        self.changed_lr = 1e-3
        self.iscl = 1
        self.cl_rate = 0.1
        self.rnn_hidden_size = 64
        self.hard_num = 10
        self.use_layer_norm = True
        self.use_pretrain = True

        self.rank_value_dir = 'data/fudanNews/rank_value.json'
        self.rank_value_dir1 = 'data/fudanNews/rank_value1.json'
        self.check_checkpoints_dir = 'checkpoints/fudanNews/check'
        self.tfidf_dir = 'data/fudanNews/ctfidf.json'
        self.words_similar_dir = 'data/fudanNews/words_similar.json'
        self.checkpoints_dir = 'checkpoints/fudanNews'
        self.raw_train_dir = 'data/fudanNews/train.txt'
        self.raw_test_dir = 'data/fudanNews/test.txt'
        self.raw_train_json_dir = 'data/fudanNews/raw_train.json'
        self.train_dir = 'data/fudanNews/train.json'
        self.new_train_dir = 'data/fudanNews/new_train.json'
        self.val_dir = 'data/fudanNews/val.json'
        self.test_dir = 'data/fudanNews/test.json'
        self.data_dir = 'data/fudanNews/data.pkl'
        self.log_level = 'info'
        self.log_filename = 'logs/log.log'
        self.token_dir = 'data/fudanNews/token.txt'
        self.label_dir = 'data/fudanNews/label.txt'
        self.raw_vocab_dir = 'data/fudanNews/raw_vocab.txt'
        self.vocab_dir = 'data/fudanNews/vocab.txt'
        self.embedding_model_dir = 'pretrain_models/sgns.sogou.word'
        self.raw_embedding_dir = 'pretrained_models/sogou.txt'
        self.embedding_dir = 'pretrain_models/GoogleNews-vectors-negative300_filtered.txt'
        self.check_dir = 'data/fudanNews/check.txt'
        self.words_all = 'data/fudanNews/words_all.txt'
        self.replace_words_dir = 'data/fudanNews/replace_words.txt'
        self.save_acc_dir = 'data/fudanNews/save_acc.json'
        self.each_acc_dir = 'data/fudanNews/each_acc.json'
        self.bert_dir = 'roberta'
        self.wrong_idx_dir = 'data/wrong_idx.json'

