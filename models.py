import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaModel, BertConfig, BertModel
import os


INIT_FUNC = {
    'uniform': uniform_,
    'kaiming_uniform': kaiming_uniform_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_normal': kaiming_normal_
}

class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 ):
        """
        embedding layer
        :param vocab_map: vocab.v2i[filed] -> Dict{Str: Int}
        :param embedding_dim: Int, config.embedding.token.dimension
        :param vocab_name: Str, 'token' or 'label'
        :param config: helper.configure, Configure Object
        :param padding_index: Int, index of padding word
        :param pretrained_dir: Str,  file path for the pretrained embedding file
        :param model_mode: Str, 'TRAIN' or 'EVAL', for initialization
        :param initial_type: Str, initialization type
        :param negative_slope: initialization config
        :param mode_fan: initialization config
        :param activation_type: None
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = torch.nn.Dropout(p=config.keep_prob)
        self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index)

        # initialize lookup table
        assert initial_type in INIT_FUNC
        if initial_type.startswith('kaiming'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=negative_slope,
                                                        mode=mode_fan,
                                                        nonlinearity=activation_type)
        elif initial_type.startswith('xavier'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        gain=torch.nn.init.calculate_gain(activation_type))
        else:
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=-0.25,
                                                        b=0.25)

        if model_mode == 'TRAIN' and config.use_pretrain \
                and pretrained_dir is not None and pretrained_dir != '':
            self.load_pretrained(embedding_dim, vocab_map, vocab_name, pretrained_dir)

        if padding_index is not None:
            self.lookup_table[padding_index] = 0.0
        self.embedding.weight.data.copy_(self.lookup_table)
        self.embedding.weight.requires_grad = True
        del self.lookup_table

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, pretrained_dir):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        print('Loading {}-dimension {} embedding from pretrained file: {}'.format(
            embedding_dim, vocab_name, pretrained_dir))
        with open(pretrained_dir, 'r', encoding='utf8') as f_in:
            num_pretrained_vocab = 0
            for line in tqdm(f_in):
                row = line.rstrip('\n').split(' ')
                if len(row) == 2:
                    assert int(row[1]) == embedding_dim, 'Pretrained dimension %d dismatch the setting %d' \
                                                         % (int(row[1]), embedding_dim)
                    continue
                if row[0] in vocab_map:
                    current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
                    self.lookup_table[vocab_map[row[0]]] = current_embedding
                    num_pretrained_vocab += 1
        print('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        print('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def forward(self, vocab_id_list):
        """
        :param vocab_id_list: torch.Tensor, (batch_size, max_length)
        :return: embedding -> torch.FloatTensor, (batch_size, max_length, embedding_dim)
        """
        embedding = self.embedding(vocab_id_list)
        return self.dropout(embedding)


class CNNClassifier(nn.Module):
    def __init__(self, cfg, vocab_map):
        super(CNNClassifier, self).__init__()

        # self.embedding = nn.Embedding(len(vocab_map), cfg.embedding_size, vocab_map['<PADDING>'])
        self.cfg = cfg
        self.embedding = EmbeddingLayer(vocab_map, cfg.embedding_size, 'token', cfg, padding_index=vocab_map['<PADDING>'],
                                        pretrained_dir=cfg.raw_embedding_dir, initial_type='uniform')

        self.convs = nn.ModuleList()
        for kernel_size in cfg.filter_sizes:
            self.convs.append(torch.nn.Conv1d(
                cfg.embedding_size,
                cfg.num_filters,
                kernel_size,
                padding=kernel_size // 2
            ))

        self.dropout = nn.Dropout(p=cfg.keep_prob)
        self.layer_norm = nn.LayerNorm(cfg.num_filters)
        self.batch_norm = nn.BatchNorm1d(cfg.num_filters)
        self.dense = nn.Linear(cfg.num_filters*len(cfg.filter_sizes), cfg.num_classes)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        embedding = embedding.transpose(1, 2)
        topk_text_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(embedding))
            topk_text = torch.topk(convolution, 1)[0].view(embedding.size(0), -1) # top k max-pooling
            if self.cfg.use_layer_norm:
                topk_text = self.layer_norm(topk_text)
            topk_text = topk_text.unsqueeze(1)
            topk_text_outputs.append(topk_text)
        text_feature = torch.cat(topk_text_outputs, 1)
        text_feature = text_feature.view(text_feature.shape[0], -1)
        feature_output = self.dropout(text_feature)
        logits = self.dense(feature_output)

        return logits, text_feature


class BertClassifier(nn.Module):
    def __init__(self, cfg):
        super(BertClassifier, self).__init__()

        self.cfg = cfg
        bert_config = BertConfig.from_json_file(os.path.join(cfg.bert_dir, 'bert_config.json'))
        self.bert = BertModel.from_pretrained(cfg.bert_dir, config=bert_config)
        # for name, param in self.bert.named_parameters():
        #     if name != 'pooler':
        #         param.requires_grad = False
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(cfg.num_filters)
        self.batch_norm = nn.BatchNorm1d(cfg.num_filters)
        self.dense = nn.Linear(768, cfg.num_classes)

    def forward(self, inputs, mask):
        last_bert_layer, pooled_output = self.bert(inputs, mask)
        last_bert_layer = last_bert_layer[:, 0, :]
        last_bert_layer = last_bert_layer.view(-1, 768)
        last_bert_layer = self.dropout(last_bert_layer)
        logits = self.dense(last_bert_layer)

        return logits, last_bert_layer


class RNNClassifier(nn.Module):
    def __init__(self, cfg, vocab_map):
        super(RNNClassifier, self).__init__()

        self.embedding = EmbeddingLayer(vocab_map, cfg.embedding_size, 'token', cfg, padding_index=vocab_map['<PADDING>'],
                                        pretrained_dir=cfg.raw_embedding_dir, initial_type='uniform')

        self.lstm = nn.LSTM(cfg.embedding_size, cfg.rnn_hidden_size, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(p=cfg.keep_prob)
        self.dense = nn.Linear(cfg.rnn_hidden_size * 2, cfg.num_classes)
        self.num_layers = 1

    def forward(self, inputs, seq_len):
        embedding = self.embedding(inputs)
        seq_len = seq_len.int()
        # sorted_seq_len, indices = torch.sort(seq_len, descending=True)
        # sorted_inputs = embedding[indices]
        x_pack = nn.utils.rnn.pack_padded_sequence(embedding, seq_len, batch_first=True, enforce_sorted=False)
        out_pack, (hidden, cell) = self.lstm(x_pack)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        logits = self.dense(hidden)

        return logits, hidden


class ResidualGatedConv1D(nn.Module):
    def __init__(self, cfg, filters, kernel_size, dilation_rate=1):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1d = nn.Conv1d(self.filters, self.filters*2, self.kernel_size, dilation=self.dilation_rate,
                                padding='same')
        self.layer_norm = nn.LayerNorm(self.filters)
        self.alpha = nn.Parameter(torch.Tensor(1))

    def forward(self, inputs, mask=None):
        inputs = inputs.transpose(1,2)
        outputs = self.conv1d(inputs)
        inputs = inputs.transpose(1, 2)
        outputs = outputs.transpose(1,2)
        gate = torch.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layer_norm(outputs)

        return inputs + self.alpha * outputs


class AttentionPooling1D(nn.Module):
    """通过加性Attention，将向量序列融合为一个定长向量
    """
    def __init__(self, cfg):
        super(AttentionPooling1D, self).__init__()
        self.k_dense = nn.Linear(
            cfg.num_filters,
            cfg.num_filters,
            bias=False,
        )
        self.o_dense = nn.Linear(cfg.num_filters, 1, bias=False)
    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.k_dense(x))
        x = self.o_dense(x)
        x = torch.softmax(x, 1)
        return torch.sum(x * inputs, 1)


class DGCNN(nn.Module):
    def __init__(self, cfg, vocab_map):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_map, cfg.embedding_size, 'token', cfg,
                                        padding_index=vocab_map['<PADDING>'],
                                        pretrained_dir=cfg.raw_embedding_dir, initial_type='uniform')
        self.rgc1 = ResidualGatedConv1D(cfg, cfg.num_filters, 3, dilation_rate=1)
        self.rgc2 = ResidualGatedConv1D(cfg, cfg.num_filters, 3, dilation_rate=2)
        self.rgc4 = ResidualGatedConv1D(cfg, cfg.num_filters, 3, dilation_rate=4)
        self.rgc8 = ResidualGatedConv1D(cfg, cfg.num_filters, 3, dilation_rate=8)
        self.ap = AttentionPooling1D(cfg)
        self.dropout = nn.Dropout(cfg.keep_prob)
        self.transform = nn.Linear(cfg.embedding_size, cfg.num_filters)
        self.dense = nn.Linear(cfg.num_filters, cfg.num_classes)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        x = self.transform(embedding)
        x = self.dropout(self.rgc1(x))
        x = self.dropout(self.rgc2(x))
        x = self.dropout(self.rgc4(x))
        x = self.dropout(self.rgc8(x))
        x = self.dropout(self.rgc1(x))
        x = self.dropout(self.rgc1(x))
        x = self.ap(x)
        drop_x = self.dropout(x)
        logits = self.dense(drop_x)

        return logits, x



