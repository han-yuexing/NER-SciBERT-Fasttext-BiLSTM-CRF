import csv
import json
import time
import torch
from torch.utils.data import DataLoader
from package.model import SciBert_FastText_BiLSTM_CRF
from package.dataset import SLSDataset, id2tag, decode_tags_from_ids
from package.metrics import Score
from torch.optim import Adam
from tqdm import tqdm
from pprint import pprint
import gensim


def logs(data, json_path):
    with open(json_path, "a", encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

fasttext = gensim.models.keyedvectors.KeyedVectors.load("fasttext/fasttext_embeddings-MINIFIED.model")
fasttext.bucket = 2000000

bert_path = 'resource/scibert-hface'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lr = 2e-3
batch_size = 20
lstm_dropout_rate = 0.4
model = SciBert_FastText_BiLSTM_CRF(bert_path, len(id2tag), bert_lstm_hidden_dim=768, w2v_lstm_hidden_dim=100, lstm_dropout_rate=lstm_dropout_rate).to(device)
model.reset_parameters()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

dataset = SLSDataset('sls/train.json', model.tokenizer, fasttext)
dataloader_train = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False, drop_last=True)

dataset = SLSDataset('sls/val.json', model.tokenizer, fasttext)
dataloader_valid = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False, drop_last=False)

rows = []
epoch = 500
file_name = 'logs-SciBERT-FastText-ep' + str(epoch) + '-bs' + str(batch_size) + '-lr0.002' + '-dropout0.4'

for _ in range(epoch):
    model.train()
    with tqdm(desc='[' + str(_+1) + '/' + str(epoch) + '] ' + 'Train', total=len(dataloader_train)) as t:
        for i, (bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings) in enumerate(dataloader_train):
            bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings = [_.to(device) for _ in (bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings)]
            optimizer.zero_grad()
            loss = model.loss(batch_size, bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings)
            loss.backward()
            optimizer.step()

            t.update(1)
            t.set_postfix(loss=float(loss))

    model.eval()
    row_list = []
    with torch.no_grad():
        score = Score()
        for i, (bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings) in enumerate(tqdm(dataloader_valid, desc='[' + str(_+1) + '/' + str(epoch) + '] ' + 'Test')):
            bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings = [_.to(device) for _ in (bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings)]
            y_pred = model(batch_size, bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings)
            y_pred = decode_tags_from_ids(y_pred)
            y_true = decode_tags_from_ids(bert_labels)
            score.update(y_pred, y_true)
        score_out = score.compute()
        logs(score_out[1], file_name + '.json')
        row_list.append(_+1)
        row_list.append(score_out[0]['f1'])
        row_list.append(score_out[0]['p'])
        row_list.append(score_out[0]['r'])
        rows.append(row_list)
        pprint(score_out[0])
    print()
    time.sleep(0.5)

headers = ['Epoch', 'F1 Score', 'Precision', 'Recall']
with open(file_name + '.csv', 'w', newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)