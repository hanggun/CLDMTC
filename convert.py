from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import codecs
from config import AGNewsConfig, CnewsConfig, FudanNewsConfig, News20Config
import json
from tqdm import tqdm
from collections import Counter


def create_words_all():
    files = [config.train_dir, config.val_dir, config.test_dir]
    counter = Counter()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                tokens = line['token']
                for word in tokens:
                    counter[word] += 1
    words, value = zip(*counter.most_common())
    with open(config.words_all, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))


def find_similarity_words():
    model = KeyedVectors.load_word2vec_format(config.embedding_model_dir, binary=False, no_header=True)
    dataset = dict()
    with codecs.open(config.words_all, 'r', encoding='utf-8') as f:
        for w in tqdm(f):
            w = w.strip()
            try:
                sim = model.most_similar(w)[:5]
                dataset[w] = sim
            except:
                continue

    with codecs.open(config.words_similar_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # config = AGNewsConfig()
    # config = FudanNewsConfig()
    # config = CnewsConfig()
    config = News20Config()
    create_words_all()
    # find_similarity_words()
    # model = KeyedVectors.load_word2vec_format('pretrained_models/sgns.sogou.word', binary=False)
    # model.save_word2vec_format('pretrained_models/sogou.txt')
