import re
import json

en_mapping = {
        '½': '1/2', '¼': '1/4', '¾': '3/4',
        '⅓': '1/3', '⅔': '2/3', '⅕': '1/5',
        '⅖': '2/5', '⅗': '3/5', '⅘': '4/5',
        '⅙': '1/6', '⅚': '5/6', '⅛': '1/8',
        '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
        'mnce': 'minced',
        'maccha': 'matcha',

    }
    
it_mapping = {
    r"panini con l'hashish": "hash browns",
    'a lisca di pesce': 'lb',
    'può': 'barattolo',
    'butter': 'burro',
    'onion': 'cipolla',
    'calce': 'lime',
    'farina di tutti i tipi': 'farina 00',
    'tutti gli usi': '00',
    'terra': 'macinato',
    'prosciugato': 'scolato',
    'hashish': 'hash browns',
    'bastoni': 'bastoncini',
    'bollito duro': 'sodo',
    'spuntato': 'scoppiato',
    'gherigli': 'chicchi',
    'sogno': 'panna',
    'trattino': 'pizzico',
    'trattini': 'pizzichi',
    'strapazzato': 'strapazzate',
    'Irlandese': 'irlandese',
    'miniature': 'mini',
    'condimento ranch': 'salsa ranch',
    'avena arrotolata': "fiocchi d'avena",
    'periodo di harina': 'farina di mais',
    'condimento mille isole': 'salsa Thousand Island',
    'seni': 'petti',
    'vaso': 'barattolo',
    'salice piccante': 'salsa piccante',
    'scuoiato': 'senza la pelle',
    "Un'oncia": "1 oncia",
    'peperoni jalapeno': 'peperoncini jalapeno',
    'mite': 'delicato',
    'drenato': 'scolato',
    'condimento per insalata mille isole': 'salsa Thousand Island per insalata',
    'pepperoncini peppers': 'peperoncini',
    'mirino': 'mirin',
    'feta cheese': 'formaggio feta',
    'massa': 'impasto',
    'spaccatura': 'con taglio',
    'dimezzato': 'tagliato in due',
    'impreparato': 'senza lievito',
    'maneggevole': 'manciata',
    'saldamente imballato': 'compatto',
    'gms': 'g',
    'magro macinato': 'macinato magro',
    'piccolo farinoso': 'piccole e farinose',
    'condimento greco universale': 'condimento greco',
    'offerte di pollo': 'filetti di pollo',
    'pelle accesa o spenta': 'con o senza pelle',
    'spruzzi': 'sprinkles',
    'spilli': 'pizzichi',
    'farina da forno': 'farina 00',
    'per cuocere il riso in': 'per cuocere il riso',
    'dragare': 'per infarinare',
    'Alfredo Salice': 'salsa Alfredo',
    'dice into 1/4" squares': 'taglia a cubetti di 0.5 cm',
    'base di pollo': 'brodo di pollo',
    'alveolare': 'senza torsolo',
    'squartato': 'tagliato in quattro',
    'vanilla essence': 'estratto di vaniglia',
    'Riso minuto': 'Minute Rice',
    'tofu morbido silken': 'tofu morbido vellutato',
    'mnce': 'macinato',
    'avanzo': 'avanzato',
    'arrosto a lama': 'arrosto di spalla',
    'pasta con il fiocco': 'farfalle',
    'coriander': 'coriandolo',
    'calorosa': 'abbondanti',
    'superfine': 'a velo',
    'teso': 'filtrato',
    'crema piena': 'intero',
    'vino di Pasqua': 'vino kosher',
    'ricotta cheese': 'ricotta',
    'potato': 'patata',
    'polpettine alla menta': 'peppermint patties',
    'trimmed': 'rifilato',
    'metà del seno': 'mezzo petto',
    'rice': 'riso',
    'spaghetti sauce': 'sugo per spaghetti',
    'vanilla pudding': 'budino alla vaniglia',
    'mozzarella cheese': 'mozzarella',
    'guarnizione montata': 'panna montata',
    'ippoglosso': 'halibut',
    'patata russa': 'patata russet',
    'borsa': 'busta',
    'arrosto di prima scelta': 'costoletta arrosto',
    'torta angelica': 'torta degli angeli',
    'Senza panna fresca': 'Cool Whip Free',
    'frusta fredda': 'Cool Whip',
    'vanilla bean': 'baccello di vaniglia',
    'date': 'datteri',
    'oro polvere di lustro': 'glitter alimentare dorato',
    'oro spruzzare': 'perline dorate',
    'La spezia': 'spezia',
    'strofinare': 'rub',
    "Pomodori d'uva": 'pomodori datterini',
    'pomodori ciliegia': 'pomodori ciliegini',
    'lance': 'bastoncini',
    'zeppe': 'spicchi',
    'pimento-stuffed': 'farcito con pimento',
    'twist al limone': 'twist di limone',
    'maccha': 'matcha',
    'bistecche a strisce': 'bistecca di filetto',
    'bistecche di conchiglia': 'bistecca di controfiletto',
    'panna pesante': 'panna fresca liquida',
    'pasta di baccelli di vaniglia': 'pasta di vaniglia',
    'cucchiaini arrotondati': 'cucchiaini',
    'Frescamente': 'appena',
    'croccante fritto': 'fritto croccante',
    'crema di pollo in brodo': 'crema di brodo di pollo',
    'in succo di frutta': 'sciroppate',
    'con il succo riservato': 'sciroppo messo da parte',
    'maraschino cherries': 'ciliegie maraschino',
    'senza steli': 'senza gambi',
    'Salsa bianca': 'besciamella',
    'luce': 'leggero',
    'metà e metà': 'panna fresca liquida',
    'condimento jamaicano jerk': 'condimento giamaicano jerk',
    'scossone': 'jerk',
    'de-seminato': 'senza semi',
    'seminato': 'senza semi',
    'artichokes': 'carciofi',
}

mappings = {
    'en': en_mapping,
    'it': it_mapping,
}

def sub_shift(json_data, mappings, *, lang):
    text_key = f'text_{lang}'
    ents_key = f'ents_{lang}'

    pattern = re.compile('|'.join(re.escape(key) for key in mappings[lang].keys()))

    for j, annotation in enumerate(json_data['annotations']):
        text = annotation[text_key]
        entities = annotation[ents_key]
        adjustment = 0

        for match in re.finditer(pattern, text):
            match_index = match.start() + adjustment
            match_contents = match.group()
            subbed_text = mappings[lang][match_contents]

            len_diff = len(subbed_text) - len(match_contents)
            text = text[:match_index] + subbed_text + text[match_index + len(match_contents):]
            # Adjust the indices of subsequent annotations
            for entity in entities:
                start, end = entity[0], entity[1]
                if start <= match_index and end > match_index:
                    entity[0] = start
                    entity[1] = end + len_diff
                if start > match_index:
                    entity[0] = start + len_diff
                    entity[1] = end + len_diff
            adjustment += len_diff
        annotation[text_key] = text.replace("⁄", "/")
        # print(j, [annotation[text_key][entities[i][0]:entities[i][1]] for i in range(len(entities))])
    return json_data

class Converter:
    def __init__(self, *, patterns) -> None:
        self.uom = None

    def str_frac_to_float(self, s):
        nums = [float(n) for n in s.split('/')]
        if len(nums)>1:
            out = (nums[0]*self.uom)/nums[1]
            if out>10:
                return int(out)
            else:
                return round(out, ndigits=1)
        out = nums[0]*self.uom
        if out>10:
            return int(out)
        else:
            return round(out, ndigits=1)

    @staticmethod
    def tighten_slash(s):
        slash_pattern = r'\s*/\s*'
        return re.sub(slash_pattern, '/', s.strip())

    def convert_num(self, input_string):
        extremes = [s for s in input_string.split('-') if s != '']
        nums = []
        for ext in extremes:
            num_buffer = 0
            int_and_frac = self.tighten_slash(ext).split()
            num_buffer = sum(self.str_frac_to_float(part) for part in int_and_frac)
            nums.append(num_buffer)
        return ' - '.join([str(n) for n in nums])
    
    def localize_ingredients(self, sample_list, lang = 'it'):
        with open('output.txt', 'w') as f:
            for j, sample in enumerate(sample_list):
                text_key = f'text_{lang}'
                ents_key = f'ents_{lang}'
                text = sample[text_key]
                ents = sample[ents_key]
                print(j, file = f)
                print(sample['text_en'], file = f)
                print([sample['text_en'][sample['ents_en'][i][0]:sample['ents_en'][i][1]] for i in range(len(sample['ents_en']))], file = f)
                print(text, file = f)
                print([text[ents[i][0]:ents[i][1]] for i in range(len(ents))], file = f)
                
                for pattern in pattern_list:
                    adjustment = 0
                    
                    regex = re.compile(pattern['pattern']) 
                    matches = re.finditer(regex, text)
                    tuple_matches = re.findall(regex, text)
                    for i, match in enumerate(matches):
                        match_text = tuple_matches[i]
                        l_strip_diff = len(match.group()) - len(match.group().lstrip())
                        match_index = match.start() + adjustment + l_strip_diff
                        if pattern['type'] == 'quantity':
                            converter.uom = pattern['ratio']
                            converted_text = self.convert_num(match_text.replace(',', '.')).replace('.', ',').replace(',0', '') + ' '
                        elif pattern['type'] == 'uom':
                            converted_text = pattern['uom']
                        elif pattern['type'] == 'plain':
                            converted_text = pattern['sub']

                        len_diff = len(converted_text) - len(match_text)
                        text = text[:match_index] + converted_text + text[match_index - l_strip_diff + len(match_text):]
                        # Adjust the indices of subsequent annotations
                        for ent in ents:
                            start, end = ent[0], ent[1]
                            if end > match_index:
                                ent[1] = end + len_diff
                            if start > match_index:
                                ent[0] = start + len_diff
                                ent[1] = end + len_diff
                        adjustment += len_diff
                sample['id'] = j
                sample[text_key] = text
                print(text, file = f)
                print([text[ents[i][0]:ents[i][1]] for i in range(len(ents))], file = f)
                print('-----------------------------------------------------------', file = f)
        return sample_list

pattern_list = [
    # fix uom
    {'pattern': r'chili pepper(?!\w)', 'type': 'plain', 'sub': 'peperoncino'},
    {'pattern': r'cancellato(?!\w)', 'type': 'plain', 'sub': 'strofinate'},
    {'pattern': r'(?<=cucchiai\s)come(?!\w)', 'type': 'plain', 'sub': 'menta'},
    {'pattern': r'come(?=\s(?:foglie|foglia))', 'type': 'plain', 'sub': 'menta'},

    {'pattern': r'(\d[-/\.\,\d\s]*)tazz.(?!\w)', 'ratio': 236, 'uom': 'g', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:(?:once fluide)|(?:oncia fluida)|once|oncia|oz\.|oz)(?!\w)', 'ratio': 28.35, 'uom': 'g', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:chili|chilo|lbs?\.?|libbre|pounds|pound|sterline|sterlina)(?!\w)', 'ratio': 0.45, 'uom': 'kg', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:di pollice|pollic.)(?!\w)', 'ratio': 2.54, 'uom': 'cm', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:pinte|pinta)(?!\w)', 'ratio': 473, 'uom': 'ml', 'type': 'quantity'},

    {'pattern': r'tazz.(?!\w)', 'ratio': 236, 'uom': 'g', 'type': 'uom'},
    {'pattern': r'(?:(?:once fluide)|(?:oncia fluida)|once|oncia|oz\.|oz)(?!\w)', 'ratio': 28.35, 'uom': 'g', 'type': 'uom'},
    {'pattern': r'(?:chili|chilo|lbs?\.?|libbre|pounds|pound|sterline|sterlina)(?!\w)', 'ratio': 0.45, 'uom': 'kg', 'type': 'uom'},
    {'pattern': r'(?:di pollice|pollic.)(?!\w)', 'ratio': 2.54, 'uom': 'cm', 'type': 'uom'},
    {'pattern': r'(?:pinte|pinta)(?!\w)', 'ratio': 473, 'uom': 'ml', 'type': 'uom'},

    {'pattern': r'tigli(?!\w)', 'type': 'plain', 'sub': 'lime'},
    {'pattern': r'orso(?!\w)', 'type': 'plain', 'sub': 'birra'},
    {'pattern': r'orecchie(?!\w)', 'type': 'plain', 'sub': 'pannocchie'},
    {'pattern': r'(?<=\d\s)misurini(?!\w)', 'type': 'plain', 'sub': 'palline'},
    {'pattern': r'marrone(?= zucchero)', 'type': 'plain', 'sub': 'di canna'},
    {'pattern': r'marrone(?= chiaro zucchero)', 'type': 'plain', 'sub': 'di canna'},
    {'pattern': r'marrone(?= scuro zucchero)', 'type': 'plain', 'sub': 'di canna'},
    {'pattern': r'abbondanza(?= sale kosher)', 'type': 'plain', 'sub': 'in abbondanza'},
    {'pattern': r'kg(?= lime)', 'type': 'plain', 'sub': 'chili'},
    {'pattern': r'(?<=mini )peperoni', 'type': 'plain', 'sub': 'peperoncini'},
]

converter = Converter(patterns=pattern_list)

json_file = './data/EW-TASTE_en-it_DEEPL.json'
with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# clean up translation mistakes by doing simple substitutions first with this function
modified_data = sub_shift(data, mappings, lang = 'en')
modified_data = sub_shift(data, mappings, lang = 'it')
localized_data = converter.localize_ingredients(modified_data['annotations'])
# save to updated json file with different name
with open(json_file[:-5] + '_localized_uom.json', 'w', encoding='utf-8') as file:
    json.dump(localized_data, file, ensure_ascii=False, indent=4)