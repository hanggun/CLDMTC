import json
from config import AGNewsConfig, CnewsConfig, FudanNewsConfig, News20Config
from spft.multiprocess import multi_process
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

agnews_cfg = AGNewsConfig()
cnews_cfg = CnewsConfig()
fudan_news_cfg = FudanNewsConfig()
news20_cfg = News20Config()
with open('../' + fudan_news_cfg.tfidf_dir, 'r', encoding='utf-8') as f:
    ctfidf = json.load(f)
def compute_rank_value(sample):
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


class ComputeRankValue:
    def __init__(self, cfg):
        self.initialized = False
        self.cfg = cfg

    def check_initialize(self):
        if not self.initialized:
            self.initialize()

    def initialize(self):
        with open('../'+self.cfg.rank_value_dir, 'r', encoding='utf-8') as f:
            self.ctfidf = json.load(f)
        self.initialized = True
        print('successfully initialized')

    def forward(self, sample):
        """计算文本中与label最相关的词，排序输出"""
        self.check_initialize()
        tokens, label = sample
        res = dict()
        new_tokens = []
        for token in tokens:
            if token not in new_tokens:
                new_tokens.append(token)
        tokens = new_tokens
        for w in tokens:
            if w in self.ctfidf:
                idf = self.ctfidf[w]['idf']
                label_dict = self.ctfidf[w]['label']
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



def save_rank_value(cfg, input_dir, output_dir):
    datasets = []
    with open('../' + input_dir, 'r', encoding='utf-8') as f:
        for idx1, line in enumerate(f):
            line = json.loads(line)
            tokens = line['token'][:cfg.maxlen]
            label = line['label']
            datasets.append((tokens, label))

    # compute = ComputeRankValue(fudan_news_cfg)
    # compute.initialize()
    rank_values = multi_process(compute_rank_value, tqdm(datasets), 6, is_queue=True)
    # with Pool(6) as pool:
    #     rank_values = pool.map(compute.forward, tqdm(datasets))
    with open('../' + output_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(rank_values, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    agnews_cfg = AGNewsConfig()
    cnews_cfg = CnewsConfig()
    fudan_news_cfg = FudanNewsConfig()
    news20_cfg = News20Config()

    # save_rank_value(news20_cfg, news20_cfg.train_dir, news20_cfg.rank_value_dir)
    # save_rank_value(agnews_cfg, agnews_cfg.train_dir, agnews_cfg.rank_value_dir)
    save_rank_value(fudan_news_cfg, fudan_news_cfg.train_dir, fudan_news_cfg.rank_value_dir)
    # save_rank_value(cnews_cfg, cnews_cfg.train_dir, cnews_cfg.rank_value_dir)