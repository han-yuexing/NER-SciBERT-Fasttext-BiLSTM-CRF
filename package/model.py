from torch import nn
from package.nn import ConditionalRandomField
from transformers import BertModel, PreTrainedTokenizerFast, BertConfig
import torch

class SciBert_FastText_BiLSTM_CRF(nn.Module):

    def __init__(self, path_to_bert, tag_set_size,  bert_lstm_hidden_dim=768, w2v_lstm_hidden_dim=100, lstm_dropout_rate=0.1, freeze_bert=True):
        super(SciBert_FastText_BiLSTM_CRF, self).__init__()
        model = BertModel.from_pretrained(path_to_bert)
        if freeze_bert:
            for param in model.parameters():
                param.requires_grad = False
        self.bert = model
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(path_to_bert)
        self.bert_biLstm = nn.LSTM(bert_lstm_hidden_dim + w2v_lstm_hidden_dim,
                                   (bert_lstm_hidden_dim + w2v_lstm_hidden_dim) // 2,
                                   num_layers=2,
                                   bidirectional=True,
                                   dropout=lstm_dropout_rate,
                                   batch_first=True)
        self.bert_hidden2tag = nn.Linear(bert_lstm_hidden_dim + w2v_lstm_hidden_dim, tag_set_size)
        self.crf = ConditionalRandomField(tag_set_size)

    def reset_parameters(self):
        self.crf.reset_parameters()


    def forward(self, batch_size, bert_input: torch.LongTensor, bert_mask: torch.ByteTensor, bert_target: torch.LongTensor, w2v_input: torch.LongTensor, w2v_mask: torch.ByteTensor, w2v_target: torch.LongTensor, mappings: torch.IntTensor):
        bert_x = self.bert(input_ids=bert_input, attention_mask=bert_mask)[0]
        bert_w2v_emb = torch.zeros(bert_x.shape[0], bert_x.shape[1], bert_x.shape[2] + w2v_input.shape[2])
        for batch_num in range(min(batch_size, bert_input.shape[0])):
            single_bert_x = bert_x[batch_num]
            single_w2v_x = w2v_input[batch_num]
            single_mappings = mappings[batch_num]
            for i, mapping in enumerate(single_mappings):
                if mapping == -1:
                    break
                else:
                    bert_w2v_emb[batch_num][i + 1] = torch.cat((single_bert_x[i + 1], single_w2v_x[mapping]), dim=0)
        bert_w2v_emb = bert_w2v_emb.to('cuda')
        bert_w2v_emb, _ = self.bert_biLstm(bert_w2v_emb)
        bert_w2v_emb = self.bert_hidden2tag(bert_w2v_emb)

        return self.crf(bert_w2v_emb, bert_mask)

    def loss(self, batch_size, bert_input: torch.LongTensor, bert_mask: torch.ByteTensor, bert_target: torch.LongTensor, w2v_input: torch.LongTensor, w2v_mask: torch.ByteTensor, w2v_target: torch.LongTensor, mappings: torch.IntTensor):
        bert_x = self.bert(input_ids=bert_input, attention_mask=bert_mask)[0]
        bert_w2v_emb = torch.zeros(bert_x.shape[0], bert_x.shape[1], bert_x.shape[2] + w2v_input.shape[2])
        for batch_num in range(batch_size):
            single_bert_x = bert_x[batch_num]
            single_w2v_x = w2v_input[batch_num]
            single_mappings = mappings[batch_num]
            for i, mapping in enumerate(single_mappings):
                if mapping == -1:
                    break
                else:
                    bert_w2v_emb[batch_num][i+1] = torch.cat((single_bert_x[i+1], single_w2v_x[mapping]), dim=0)
        bert_w2v_emb = bert_w2v_emb.to('cuda')
        bert_w2v_emb, _ = self.bert_biLstm(bert_w2v_emb)
        bert_w2v_emb = self.bert_hidden2tag(bert_w2v_emb)
        return self.crf.neg_log_likelihood_loss(bert_w2v_emb, bert_mask, bert_target)
