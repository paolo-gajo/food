import json
from tqdm.auto import tqdm
from evaluate import load
from icecream import ic
import os
import logging
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import get_entities_from_sample, get_relations_from_sample

def word_idx_to_span(idx, wordlist):
    left = len(''.join(wordlist[:idx])) + len(wordlist[:idx])
    right = left + len(wordlist[idx])
    return {'start': left, 'end': right}

def match_id(ent_src, entity_list, lang_tgt = 'it'):
    ents = [ent for ent in entity_list if ent['type'] == 'labels' and ent[f'from_name'] == f'label_{lang_tgt}']
    rels = [rel for rel in entity_list if rel['type'] == 'relation']
    from_id = ''
    for rel in rels:
        if rel['to_id'] == ent_src['id']:
            from_id = rel['from_id']
            break
    if from_id:
        for ent in ents:
            if ent['id'] == from_id:
                return ent
    else:
        return {'id': ''}

def format_ls_ent(from_name='', id='', to_name='', type='', start=0, end=0, labels=''):
    ent_ls = {
                    'from_name': from_name,
                    'id': id,
                    'to_name': to_name,
                    'type': type,
                    'value': {
                        'start': start,
                        'end': end,
                        'labels': [labels],
                        }
                    }
    return ent_ls

def pharaoh_aggregate_targets(pharaoh_alignments_line):
    '''
    Given a pharaoh-style line of alignments, e.g.,
    '0-0 1-1 1-2 2-3 3-4 4-5 5-6 5-7 6-8 7-9 8-10'
    aggregate target words aligned to the same source word
    '''
    line_alignments_tuples = [tuple([int(a) for a in line_alignment.split('-')]) for line_alignment in pharaoh_alignments_line.split()]
    alignment_src_last = -1
    alignments_aggregated = []
    alignments_tgt = []
    for alignment in line_alignments_tuples + [(-1,-1)]: # add a dummy tuple to handle the last append
        if alignment[0] != alignment_src_last and alignments_tgt:
            # check if the source entity is different from the last
            # check if we have alignments loaded into alignments_tgt
            alignments_aggregated.append({'source': alignment_src_last, 'target': alignments_tgt})
            alignments_tgt = []
            alignments_tgt.append(alignment[1])
        else:
            alignments_tgt.append(alignment[1])
        alignment_src_last = alignment[0]
    return alignments_aggregated

def flatten_list(input_list):
    input_list = [el for sublist in input_list for el in sublist]
    return input_list

def get_minmax(dict_list):
    values = []
    for el in dict_list:
        values.extend(el.values())
    start = min(values)
    end = max(values)
    return start, end

def match_src_words_to_entities(sample, alignment_list, lang_src = 'en', lang_tgt = 'it', field_name = 'ingredients', model_name = ''):
    '''
    Aggregate source words falling within the same entity
    '''
    ent_src_list = get_entities_from_sample(sample, lang=lang_src, sort=True)
    ent_tgt_list = get_entities_from_sample(sample, lang=lang_tgt, sort=True)
    relations = get_relations_from_sample(sample)
    ingredients_list_src = sample['data'][f'{field_name}_{lang_src}'].split()
    ingredients_list_tgt = sample['data'][f'{field_name}_{lang_tgt}'].split()
    alignments_tgt_final = []
    sample['predictions'].append({'model_version': model_name, 'result': [],})
    for ent_src in ent_src_list:
        start_src = ent_src['value']['start']
        end_src = ent_src['value']['end']
        label_src = ent_src['value']['labels'][0]
        for alignment in alignment_list:
            word_src = ingredients_list_src[alignment['source']]
            alignment_src_span = word_idx_to_span(alignment['source'], ingredients_list_src)
            if start_src <= alignment_src_span['start'] and alignment_src_span['end'] <= end_src:
                alignment_tgt_tmp = [word_idx_to_span(al, ingredients_list_tgt) for al in alignment['target']]
                alignments_tgt_final.extend(alignment_tgt_tmp)
                alignment_tgt_tmp = []

        if alignments_tgt_final:
            start_tgt, end_tgt = get_minmax(alignments_tgt_final)
        else:
            start_tgt = start_src
            end_tgt = end_src
        ent_tgt = match_id(ent_src, sample['annotations'][0]['result'])
        ent_it_new = format_ls_ent(from_name=f'label_{lang_tgt}', id=ent_tgt['id'], to_name=f'{field_name}_{lang_tgt}_ref', type='labels', labels=label_src,
                                       start=start_tgt,
                                       end=end_tgt,
                                       )
        sample['predictions'].append(ent_it_new)
        alignments_tgt_final = []
    return sample


def main():
    data_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced_tok_moses.json'
    # data_path = 'data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_unaligned_spaced_tok.json'

    with open(data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    # filename_alignments = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced_tok_moses_en-it_align_fast-align.txt'
    filename_alignments = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced_tok_moses_align_gpp.txt'

    # filename_alignments = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/fast-align/data/SW-TASTE_en-it_DEEPL_unaligned_spaced_align_fast-align.txt'
    # filename_alignments = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/fast-align/data/SW-TASTE_en-it_DEEPL_unaligned_spaced_align_gpp.txt'

    model_name = filename_alignments.split('_')[-1].split('.')[0]
    with open(filename_alignments, 'r', encoding='utf8') as f:
        alignments = f.readlines()

    dataset_aligned = []
    for sample, alignment in tqdm(zip(data, alignments), total=len(data)):
        aggregated_alignments = pharaoh_aggregate_targets(alignment)
        matched_sample = match_src_words_to_entities(sample, aggregated_alignments, model_name=model_name)
        dataset_aligned.append(matched_sample)

    with open(data_path.replace('.json', f'_preds_{model_name}.json'), 'w', encoding='utf8') as f:
        json.dump(dataset_aligned, f, ensure_ascii = False)

if __name__ == "__main__":
    main()

# metric = load("squad_v2")
# results_path = f"/home/pgajo/food/results/alignment/test/{data_path.split('/')[-1].split('.')[0]}"

# if not os.path.exists(results_path):
#     os.makedirs(results_path)


# preds = []
# trues = []

# with open(os.path.join(results_path, model_name), 'w', encoding='utf8') as f:
#     json.dump(test_metrics, f, ensure_ascii = False)

        # if eval:
        #     ent_gold = match_id(ent, line['annotations'][0]['result'])
        #     if ent_gold['id']:
        #         dict_true = {
        #             'answers': {
        #                 'answer_start': [ent_gold['value']['start']],
        #                 'text': [line['data'][f'{field_name}_{lang_tgt}'][ent_gold['value']['start']:ent_gold['value']['end']]],
        #                 },
        #                 'id': ent_gold['id']
        #             }
        #         trues.append(dict_true)
        #         dict_pred = {
        #             'prediction_text': line['data'][f'{field_name}_{lang_tgt}'][alignment_tgt_left:alignment_tgt_right],
        #             'id': ent_gold['id'],
        #             'no_answer_probability': 0,
        #         }
        #         preds.append(dict_pred)
        # results = metric.compute(predictions=preds, references=trues)