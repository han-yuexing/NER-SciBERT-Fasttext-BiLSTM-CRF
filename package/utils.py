import json
import torch

def read_train_json(path):
    """
    Read json file in SLS dataset
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                text_type_list = text.split()
                label = ['O'] * len(text_type_list)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert text[start_index:end_index + 1] == sub_name
                                sub_name_type_list = sub_name.split()
                                index_to_id = {}
                                id = 0
                                for i, s in enumerate(text):
                                    if s == ' ':
                                        id = id + 1
                                        continue
                                    index_to_id[i] = id
                                loc = text.find(sub_name)
                                begin = index_to_id[loc]
                                end = index_to_id[loc] + len(sub_name_type_list)-1
                                label[begin] = 'B-' + key
                                label[begin + 1:end + 1] = ['I-' + key] * (len(sub_name_type_list) - 1)
                if len(text_type_list) == len(label):
                    lines.append({"text": text, "label": label})
            except Exception as e:
                continue
    return lines


def decode_bio_tags(tags):
    """
    decode entity (type, start, end) from BIO style tags
    """
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(tags):

        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)

            chunk = [-1, -1, -1]
            chunk[0] = tag.split('-')[1]
            chunk[1] = i
            chunk[2] = i + 1
            if i == len(tags) - 1:
                chunks.append(chunk)

        elif tag.startswith('I-') and chunk[1] != -1:
            t = tag.split('-')[1]
            if t == chunk[0]:
                chunk[2] = i + 1

            if i == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks
