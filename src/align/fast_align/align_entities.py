import json
from tqdm.auto import tqdm

# data_to_align = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/fast-align/data/SW-TASTE_en-it_DEEPL_unaligned_spaced_fast-align_TS.json'
data_to_align = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced_TS_fa.json'

with open(data_to_align, 'r', encoding='utf8') as f:
    dataset_unaligned = json.load(f)

filename_alignments = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD-NER-ALIGN_105_spaced_TS_fa.txt'

with open(filename_alignments, 'r', encoding='utf8') as f:
    alignments = f.readlines()

def word_idx_to_span(idx, wordlist):
    left = len(''.join(wordlist[:idx])) + len(wordlist[:idx])
    right = left + len(wordlist[idx])
    return (left, right)

def flatten_list(input_list):
    input_list = [el for sublist in input_list for el in sublist]
    return input_list

def align_dataset_fast_align(data, alignments):
    '''
    args:
        data:
            Unaligned list of dicts which needs to be aligned.
        alignments:
            Pharaoh-style list of strings.

    example input args:
        data = [{
            "ingredients_en": "5 ounces rum ; 4 ounces triple sec ; 3 ounces Tia Maria ; 20 ounces orange juice",
            "ents_en": [
                [0, 1, "QUANTITY"],
                [2, 8, "UNIT"],
                [9, 12, "FOOD"],
                [15, 16, "QUANTITY"],
                [17, 23, "UNIT"],
                [24, 34, "FOOD"],
                [37, 38, "QUANTITY"],
                [39, 45, "UNIT"],
                [46, 55, "FOOD"],
                [58, 60, "QUANTITY"],
                [61, 67, "UNIT"],
                [68, 80, "FOOD"]
            ],
            "ingredients_it": "5 once di rum ; 4 once di triple sec ; 3 once di Tia Maria ; 20 once di succo d ' arancia",
            "ents_it": []
        },
        ...]

        alignments = ['0-0 1-1 1-2 2-3 3-4 4-5 5-6 5-7 6-8 7-9 8-10 9-11 10-12 10-13 11-14 11-15 13-16 14-17 15-18 13-19 17-20 16-21 16-22 16-23',
                        ...]
    '''
    aligned_dataset = []
    for i, line in tqdm(enumerate(data), total = len(data)):
        line_alignments_tuples = [tuple([int(a) for a in line_alignment.split('-')]) for line_alignment in alignments[i].split()]
        ingredients_list_en = line['ingredients_en'].split()
        ingredients_list_it = line['ingredients_it'].split()
        alignment_src_last = -1
        alignments_src_combined = []
        alignments_tgt = []

        # aggregate target words aligned to the same source word
        for alignment in line_alignments_tuples:
            # print(f'[{alignment[0]}]{word_idx_to_span(alignment[0], ingredients_list_en)}-{word_idx_to_span(alignment[1], ingredients_list_it)}: {ingredients_list_en[alignment[0]]} --> {ingredients_list_it[alignment[1]]}')
            # print('----------')
            if alignment[0] != alignment_src_last and alignments_tgt:
                alignments_src_combined.append([[alignment_src_last], alignments_tgt])
                alignments_tgt = []
                alignments_tgt.append(alignment[1])
            else:
                alignments_tgt.append(alignment[1])
            alignment_src_last = alignment[0]
        if alignments_tgt:
            alignments_src_combined.append([[alignment_src_last], alignments_tgt])

        alignments_tgt_final = []

        ents_it = []
        
        # aggregate source words falling within the same entity
        for ent in line['ents_en']:
            for alignment in alignments_src_combined:
                alignment_src_span = word_idx_to_span(alignment[0][0], ingredients_list_en)
                if alignment_src_span[0] >= ent[0] and alignment_src_span[1] <= ent[1]:
                    alignment_tgt_tmp = [word_idx_to_span(al, ingredients_list_it) for al in alignment[1]]
                    alignments_tgt_final.extend(alignment_tgt_tmp)
                    alignment_tgt_tmp = []
            if alignments_tgt_final:
                alignments_tgt_final_flat = flatten_list(alignments_tgt_final)
                alignment_tgt_left = min(alignments_tgt_final_flat)
                alignment_tgt_right = max(alignments_tgt_final_flat)
                ents_it.append([alignment_tgt_left, alignment_tgt_right, ent[2]])
            else:
                ents_it.append(ent)
            alignments_tgt_final = []
        
        aligned_dataset.append({
            'ingredients_en': line['ingredients_en'],
            'ents_en': line['ents_en'],
            'ingredients_it': line['ingredients_it'],
            'ents_it': ents_it,
        })
    return aligned_dataset

dataset_aligned = align_dataset_fast_align(dataset_unaligned, alignments)
with open(data_to_align.replace('.json', '_aligned.json'), 'w', encoding='utf8') as f:
    json.dump(dataset_aligned, f, ensure_ascii = False)