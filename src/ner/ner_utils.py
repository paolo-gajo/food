import re
from tqdm.auto import tqdm

# def make_ner_sample(sample,
#                     tokenizer,
#                     label_dict,
#                     lang = 'en',
#                     text_name = 'text',
#                     label_field = 'predictions',
#                     label_list = []
#                     ):
#     sample_text = sample['data'][f'{text_name}_{lang}']
#     tokens = tokenizer(sample_text,
#                     #    return_tensors='pt',
#                        truncation = True,
#                        padding = 'max_length'
#                        )
#     # print(tokens)
#     labels = []
#     previous_match_span = (-1, -1)
#     for i, id in enumerate(tokens['input_ids']):
#         span = tokens.token_to_chars(i)
#         if int(id) not in [101, 102, tokenizer.pad_token_id]:
#             start = span.start
#             end = span.end
#             new_item = ''
#             for result in sample[label_field][0]['result']:
#                 # ic(start, end)
#                 # ic(result)
#                 if result['value']:
#                     label = result['value']['labels'][0]
#                     if start >= result['value']['start'] \
#                     and end <= result['value']['end'] \
#                     and result['from_name'] == f'label_{lang}' \
#                     and re.search(re.compile(r'[.,;]'), tokenizer.decode(id)) is None \
#                     and label in label_list:
#                         if result['value']['start'] == previous_match_span[0] \
#                         and result['value']['end'] == previous_match_span[1]:
#                             new_item = 'I-' + label
#                         else:
#                             new_item = 'B-' + label
#                         labels.append(new_item)
#                         previous_match_span = (result['value']['start'], result['value']['end'])
#             if not new_item:
#                 labels.append('O')
#             # ic(i, int(id), new_item, sample['data']['text_en'][start:end], tokenizer.decode(id))
#         else:
#             new_item = -100
#             labels.append(new_item)
#     labels = [int(label_dict[label] if isinstance(label, str) else label) for label in labels]
#     tokens.update({'labels': labels})
#     return tokens

def get_ner_classes(data=None, label_field='predictions', raw_labels=None):
    if not raw_labels:
        raw_labels = []
        for line in tqdm(data['train']):
            # print(line)
            for prediction in line[label_field]:
                for result in prediction['result']:
                    if result['value']:
                        if result['value']['labels'][0] not in raw_labels:
                            raw_labels.append(result['value']['labels'][0])

    label_list = ['O']
    prefixes = [
        # '',
        'B-',
        'I-'
        ]
    for raw_label in raw_labels:
        for prefix in prefixes:
            label_list.append(prefix+raw_label)
    
    label2id = {k: v for v, k in enumerate(label_list)}
    id2label = {v: k for k, v in zip(label2id.keys(), label2id.values())}
    
    return raw_labels, label_list, label2id, id2label

def make_ner_sample(sample,
                    tokenizer,
                    label_dict,
                    lang = 'en',
                    text_name = 'text',
                    label_field = 'predictions',
                    label_list = [],
                    max_length = 512
                    ):
    
    input_ids = [101]
    attention_mask = [1]
    labels = [-100]

    ents = []

    for result in sample[label_field][0]['result']:
        if 'from_name' in result.keys()\
        and result['from_name'] == f'label_{lang}':
            result_start = result['value']['start']
            result_end = result['value']['end']
            result_label = result['value']['labels'][0]
            ents.append([result_start, result_end, result_label])
    ents.sort()
    ents_filled = []
    ents_blanks = []
    for i in range(len(ents)):
        ents_filled.append(ents[i])
        ent_blank = [ents[i-1][1], ents[i][0], 'O']
        ents_blanks.append(ent_blank)
    ents_blanks.sort()
    ents_filled.extend(ents_blanks[:-1])
    ents_filled.sort()
    
    for ent in ents_filled:
        sample_text = sample['data'][f'text_{lang}']
        ent_text = sample_text[ent[0]:ent[1]]
        ent_text_encoded = tokenizer(ent_text)
        ent_input_ids = ent_text_encoded['input_ids'][1:-1]
        ent_labels = []
        for i in range(len(ent_input_ids)):
            if ent[2] != 'O' and ent[2] in label_list:
                if i == 0:
                    ent_label = 'B-' + ent[2]
                    ent_labels.append(ent_label)
                else:
                    ent_label = -100
                    ent_labels.append(ent_label)
            else:
                ent_labels.append('O')
        input_ids.extend(ent_input_ids)
        attention_mask.extend([1 for _ in range(len(ent_input_ids))])
        labels.extend(ent_labels)
    
    input_ids.append(102)
    attention_mask.append(1)
    labels.append(-100)
    
    input_ids.extend([tokenizer.pad_token_id for _ in range(max_length - len(labels))])
    attention_mask.extend([0 for _ in range(max_length - len(labels))])
    labels = [int(label_dict[label] if isinstance(label, str) else label) for label in labels]
    labels.extend([-100 for _ in range(max_length - len(labels))])
    token_type_ids = [0 for _ in range(max_length)]

    entry = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'token_type_ids': token_type_ids,
    }

    return entry