import re
import json
import random
import sys
sys.path.append('/home/pgajo/food/src')
from utils import label_studio_to_tasteset, tasteset_to_label_studio, TASTEset

def randomizer(s_list):
    i = random.randrange(len(s_list))
    return s_list[i]

en_mapping = {
    '½': '1/2', '¼': '1/4', '¾': '3/4',
    '⅓': '1/3', '⅔': '2/3', '⅕': '1/5',
    '⅖': '2/5', '⅗': '3/5', '⅘': '4/5',
    '⅙': '1/6', '⅚': '5/6', '⅛': '1/8',
    '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
    # ';': ' ; ',
    # '  ;  ': ' ',
    # ' ;  ': ' '
    # 'mnce': 'minced',
    # 'maccha': 'matcha',
    # 'bushes baked beans': "Bush's Baked Beans",
    # '/colored': 'colored',
    # '2 2 ': '2 ',
    # 'empire apples or': 'empire apples',
    # 'lipton recip secret': 'Lipton Recipe Secrets',
    # 'matzoh': 'matzah',
    # 'course': 'coarse',
    # 'rotel': 'Rotel',
    }

es_mapping = {
    '½': '1/2', '¼': '1/4', '¾': '3/4',
    '⅓': '1/3', '⅔': '2/3', '⅕': '1/5',
    '⅖': '2/5', '⅗': '3/5', '⅘': '4/5',
    '⅙': '1/6', '⅚': '5/6', '⅛': '1/8',
    '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
    # ';': ' ; ',
    # '  ;  ': ' ',
    # ' ;  ': ' '
    }

de_mapping = {
    # ';': ' ; ',
    # '  ;  ': ' ',
    # ' ;  ': ' '
    }

it_mapping = {
    # r"panini con l'hashish": "hash browns",
    # 'a lisca di pesce': 'lb',
    # 'può': 'barattolo',
    # 'butter': 'burro',
    # 'onion': 'cipolla',
    # 'calce': 'lime',
    # 'farina di tutti i tipi': 'farina 00',
    # 'tutti gli usi': '00',
    # 'terra': 'macinato',
    # 'prosciugato': 'scolato',
    # 'hashish': 'hash browns',
    # 'bastoni': 'bastoncini',
    # 'bollito duro': 'sodo',
    # 'spuntato': 'scoppiato',
    # 'gherigli': 'chicchi',
    # 'sogno': 'panna',
    # 'trattino': 'pizzico',
    # 'trattini': 'pizzichi',
    # 'strapazzato': 'strapazzate',
    # 'Irlandese': 'irlandese',
    # 'miniature': 'mini',
    # 'condimento ranch': 'salsa ranch',
    # 'avena arrotolata': "fiocchi d'avena",
    # 'periodo di harina': 'farina di mais',
    # 'condimento mille isole': 'salsa Thousand Island',
    # 'seni': 'petti',
    # 'vaso': 'barattolo',
    # 'salice piccante': 'salsa piccante',
    # 'scuoiato': 'senza la pelle',
    # "Un'oncia": "1 oncia",
    # 'peperoni jalapeno': 'peperoncini jalapeno',
    # 'mite': 'delicato',
    # 'drenato': 'scolato',
    # 'condimento per insalata mille isole': 'salsa Thousand Island per insalata',
    # 'pepperoncini peppers': 'peperoncini',
    # 'mirino': 'mirin',
    # 'feta cheese': 'formaggio feta',
    # 'massa': 'impasto',
    # 'spaccatura': 'con taglio',
    # 'dimezzato': 'tagliato in due',
    # 'impreparato': 'senza lievito',
    # 'maneggevole': 'manciata',
    # 'saldamente imballato': 'compatto',
    # 'gms': 'g',
    # 'magro macinato': 'macinato magro',
    # 'piccolo farinoso': 'piccole e farinose',
    # 'condimento greco universale': 'condimento greco',
    # 'offerte di pollo': 'filetti di pollo',
    # 'pelle accesa o spenta': 'con o senza pelle',
    # 'spruzzi': 'sprinkles',
    # 'spilli': 'pizzichi',
    # 'farina da forno': 'farina 00',
    # 'per cuocere il riso in': 'per cuocere il riso',
    # 'dragare': 'per infarinare',
    # 'Alfredo Salice': 'salsa Alfredo',
    # 'dice into 1/4" squares': 'taglia a cubetti di 0.5 cm',
    # 'base di pollo': 'brodo di pollo',
    # 'alveolare': 'senza torsolo',
    # 'squartato': 'tagliato in quattro',
    # 'vanilla essence': 'estratto di vaniglia',
    # 'Riso minuto': 'Minute Rice',
    # 'tofu morbido silken': 'tofu morbido vellutato',
    # 'mnce': 'macinato',
    # 'avanzo': 'avanzato',
    # 'arrosto a lama': 'arrosto di spalla',
    # 'pasta con il fiocco': 'farfalle',
    # 'coriander': 'coriandolo',
    # 'calorosa': 'abbondanti',
    # 'superfine': 'a velo',
    # 'teso': 'filtrato',
    # 'crema piena': 'intero',
    # 'vino di Pasqua': 'vino kosher',
    # 'ricotta cheese': 'ricotta',
    # 'potato': 'patata',
    # 'polpettine alla menta': 'peppermint patties',
    # 'trimmed': 'rifilato',
    # 'metà del seno': 'mezzo petto',
    # 'rice': 'riso',
    # 'spaghetti sauce': 'sugo per spaghetti',
    # 'vanilla pudding': 'budino alla vaniglia',
    # 'mozzarella cheese': 'mozzarella',
    # 'guarnizione montata': 'panna montata',
    # 'ippoglosso': 'halibut',
    # 'patata russa': 'patata russet',
    # 'patate russe': 'patate russet',
    # 'borsa': 'busta',
    # 'arrosto di prima scelta': 'costoletta arrosto',
    # 'torta angelica': 'torta degli angeli',
    # 'Senza panna fresca': 'Cool Whip Free',
    # 'frusta fredda': 'Cool Whip',
    # 'vanilla bean': 'baccello di vaniglia',
    # 'date': 'datteri',
    # 'oro polvere di lustro': 'glitter alimentare dorato',
    # 'oro spruzzare': 'perline dorate',
    # 'La spezia': 'spezia',
    # 'strofinare': 'rub',
    # "Pomodori d'uva": 'pomodori datterini',
    # 'pomodori ciliegia': 'pomodori ciliegini',
    # 'lance': 'bastoncini',
    # 'zeppe': 'spicchi',
    # 'pimento-stuffed': 'farcito con pimento',
    # 'twist al limone': 'twist di limone',
    # 'maccha': 'matcha',
    # 'bistecche a strisce': 'bistecca di filetto',
    # 'bistecche di conchiglia': 'bistecca di controfiletto',
    # 'panna pesante': 'panna fresca liquida',
    # 'pasta di baccelli di vaniglia': 'pasta di vaniglia',
    # 'cucchiaini arrotondati': 'cucchiaini',
    # 'Frescamente': 'appena',
    # 'croccante fritto': 'fritto croccante',
    # 'crema di pollo in brodo': 'crema di brodo di pollo',
    # 'in succo di frutta': 'sciroppate',
    # 'con il succo riservato': 'sciroppo messo da parte',
    # 'maraschino cherries': 'ciliegie maraschino',
    # 'senza steli': 'senza gambi',
    # 'Salsa bianca': 'besciamella',
    # 'luce': 'leggero',
    # 'metà e metà': 'panna fresca liquida',
    # 'condimento jamaicano jerk': 'condimento giamaicano jerk',
    # 'scossone': 'jerk',
    # 'de-seminato': 'senza semi',
    # 'seminato': 'senza semi',
    # 'artichokes': 'carciofi',
    # 'schizzo': 'goccio',
    # 'involucri integrali': 'piadine integrali',
    # 'seno': 'petto',
    # 'gambero tigre': 'gamberi giganti',
    # 'gamberi jumbo': 'gamberoni',
    # 'cilantro leaves': 'foglie di coriandolo',
    # 'miscela': 'preparato',
    # 'Miscela': 'preparato',
    # 'pizza dough': 'impasto per pizza',
    # 'salsa al pesto': 'pesto',
    # 'briciola di pane': 'pangrattato',
    # 'provolone cheese': 'provolone',
    # 'ricordato': 'decorticato',
    # 'bacchette': 'cosce',
    # 'pelle su': 'con la pelle',
    # 'per il sapore': 'per insaporire',
    # 'latte di riso normale': 'latte di riso',
    # 'rotoli di sottomarino': 'filoncini',
    # 'per la guarnizione': 'per guarnire',
    # 'prodotta': 'infuso',
    # 'lampone a scatto': 'Schnapps al lampone',
    # 'cuneo': 'spicchio',
    # 'pasta biscotto': 'impasto per biscotti',
    # 'prezzemolo in scaglie': 'fiocchi di prezzemolo',
    # 'mix per budino': 'preparato per budino',
    # 'sodio ridotto': 'a basso contenuto di sodio',
    # 'succo di vongola': 'brodo di vongole',
    # 'vanilla sugar': 'zucchero vanigliato',
    # 'tagliato a scaglie': 'tagliato a strisce',
    # 'accorciamento': 'grasso alimentare',
    # 'crema di formaggio': 'formaggio spalmabile',
    # 'tagliato del grasso': 'grasso rimosso',
    # 'Una grande pentola di': 'una pentola grande',
    # 'Avena Quaker': 'fiocchi di avena Quaker',
    # 'cespugli di fagioli al forno': "Bush's Baked Beans",
    # 'rotolo di pane francese': 'panino',
    # 'pane francese': 'pane',
    # 'Pane francese': 'pane',
    # 'EVOO': "olio extravergine di oliva",
    # 'caramelle fuse': 'candy melts',
    # 'verdura di mare': 'alghe',
    # 'umeboshi vinegar': 'aceto umeboshi',
    # '/colorato': 'colorati',
    # 'riservato': 'messo da parte',
    # 'zucchinis': 'zucchine',
    # 'grande valore': 'grande risparmio',
    # 'Angostura bitters': 'angostura',
    # 'caffè tequila': 'tequila al caffè',
    # 'grappa di butterscotch': 'schnapps al butterscotch',
    # 'cile': 'peperoncino',
    # 'pasta per rotini': 'fusilli',
    # 'più il succo': 'con il succo',
    # 'maraschino cherry juice': 'succo di ciliegia maraschino',
    # 'maraschino cherry': 'ciliegia maraschino',
    # 'Io sono lo yogurt': 'yogurt di soia',
    # 'ciliegie in succo': 'ciliegie sciroppate',
    # 'tomatillos': 'tomatillo',
    # 'vinaigrette al balsamico': "vinaigrette all'aceto balsamico",
    # 'distrutto': 'schiacciato',
    # 'a piacere': ['q.b.', 'quanto basta'],
    # 'A piacere': ['q.b.', 'quanto basta'],
    # 'secondo necessità': ['q.b.', 'quanto basta'],
    # '2 2 ': '2 ',
    # "mele dell'impero o": 'mele empire',
    # 'tagliare': 'tagliato',
    # 'tagliato con le forbici in piccoli brandelli': 'tagliato con le forbici in strisce sottili',
    # 'rasato': 'a scaglie',
    # 'sciroppo semplice': 'sciroppo',
    # 'fungo': 'funghi',
    # 'spruzzare lo spray da cucina': 'spruzzi di spray da cucina',
    # 'Condimento Old Bay': 'Old Bay Seasoning',
    # 'broccoli florets': 'cime di broccolo',
    # 'lipton recip secret': 'Lipton Recipe Secrets',
    # 'orzo pasta': 'pasta orzo',
    # 'ruota': 'fetta',
    # 'sprig': 'ramoscello',
    # 'cime di collardo': 'cavolo nero',
    # 'bistecche di fianchetto': 'bavetta',
    # 'bistecca di fianchetto': 'bavetta',
    # 'gocce da forno': 'gocce di cioccolato',
    # 'di dimensioni taco': 'di dimensioni per taco',
    # 'rondelle di 1/8': 'rondelle di 3 mm',
    # 'pizza crust': 'base per pizza',
    # 'matzoh': 'matzah',
    # 'tomato': 'pomodoro',
    # 'ditone': 'patate fingerling',
    # 'mozzarella ball': 'ciliegine di mozzarella',
    # 'mascarpone cheese': 'mascarpone',
    # 'top verde acceso o spento': 'con o senza foglie',
    # 'cola dietetica': 'cola light',
    # 'colatura di pancetta': 'grasso di pancetta',
    # 'graniglia': 'semolino',
    # 'Spessore di 1/2".': 'spessa 1 cm',
    # 'cereali di riso': 'crema di riso',
    # 'biscotti di wafer': 'wafer',
    # 'sgranato': 'sgusciato',
    # '1 giorno di vita': 'del giorno prima',
    # 'elaborato': 'a pasta fusa',
    # 'bistecca di manzo': 'sottofesa di manzo',
    # 'granatina': 'sciroppo di granatina',
    # 'mozzarella garlic bread': "pane all'aglio e mozzarella",
    # "polpetta di salsiccia": "hamburger di salsiccia",
    # 'cialde': 'waffle',
    # 'insalata di cavolo': 'insalata coleslaw',
    # 'non sbiancato': 'non sbiancata',
    # 'attivo ad alzata rapida': 'attivo a rapida lievitazione',
    # 'tazze di burro di arachidi': 'Peanut Butter Cups',
    # 'crosta di torta': 'pasta frolla',
    # 'olio di semi di vinacciolo': 'olio di vinaccioli',
    # 'tagliati ad anelli sottili': 'tagliata ad anelli sottili',
    # 'scafi rimossi': 'senza stoloni',
    # 'filetti di dentice': 'filetti di lutiano',
    # 'Salice Tabasco': 'salsa Tabasco',
    # 'farina normale': 'farina 00',
    # 'sgocciolamento': 'grasso',
    # 'grattugiato finemente fresco': 'grattugiato finemente appena',
    # 'grattugiato fresco': 'grattugiato appena',
    # 'olio di noce': 'olio di noci',
    # 'Brindisi texano': 'toast texano',
    # 'Formaggio misto messicano': 'mix di formaggio messicano',
    # 'mix per torta gialla': 'preparato per pan di Spagna',
    # 'giallo impasto per torte': 'preparato per pan di Spagna',
    # '3 cm (3/4 pollici)': '3/4 pollici',
    # 'salsa alla marinara': 'salsa marinara',
    # 'colatura': 'grasso',
    # 'scavato': 'senza stoloni',
    # 'scoop': 'palline',
    # 'pelli rimosse': 'senza pelle',
    # 'metà cotte al forno e vuote di patate': 'metà di patate cotte al forno e svuotate',
    # 'taglio spesso': 'tagliata spesso',
    # '")")': ' ")',
    # 'ditta': 'sodi',
    # 'ripieno di torta di ciliegie': 'ripieno per torta di ciliegie',
    # "farina d'avena": "porridge",
    # 'crisp rice cereal': 'riso soffiato croccante',
    # 'grano corto': 'a chicco corto',
    # 'Grana media': 'a chicco medio',
    # 'Pomodori rom': 'pomodori Roma',
    # "per l'ingrassaggio": 'per ungere la padella',
    # 'stagionata': 'condito',
    # "sottaceti all'aneto": "cetrioli sott'aceto all'aneto",
    # '4 pollici (10 cm)': '4 pollici',
    # 'ruggine patate': 'russet patate',
    # 'incrinato': 'schiacciato',
    # 'lampadina': 'testa',
    # 'club soda': 'acqua tonica',
    # 'cola soda': 'bibita cola',
    # 'spaccato longitudinalmente': 'tagliato in lunghezza',
    # 'votato': 'senza picciolo',
    # 'cucinare le mele': 'mele',
    # 'apple pie spice': 'spezie per torta di mele',
    # "l'olio della carne": "grasso della carne",
    # "pacchetto": 'confezione',
    # 'snocciolato': 'denocciolato',
    # 'pelle attaccata': 'con la pelle',
    # 'di polimerizzazione': 'per stagionatura',
    # 'Arancione': 'arancia',
    # 'arancione': 'arancia',
    # 'zestato': 'grattuggiata',
    # 'spezzato in circa 4 pezzi': 'rotto in circa 4 pezzi',
    # 'razzo': 'rucola',
    # 'gara': 'in cristalli',
    # 'rotel': 'Rotel',
    # 'come guarnizione': 'come condimento',
    # 'Avvolgimento': 'crepes',
    # 'pasta di pane': 'impasto per il pane',
    # 'pacchetti': 'confezioni',
    # 'budino3': 'budino 3',
    # 'torta a forma di pagnotta': 'plumcake',
    # 'ambito di applicazione': 'pallina',
    # 'bevanda analcolica alla cola': 'bevanda alla cola',
    # 'Da 2 1/2 a 1,4': 'Da 2 1/2 a 3',
    # 'bone in': 'con le ossa',
    # 'Zia Maria': 'Tia Maria',
    # 'Grasso ridotto': 'a basso contenuto di grassi',
    # ';': ' ; ',
    # '  ;  ': ' ',
    # ' ;  ': ' '
        
}

mappings = {
    'en': en_mapping,
    'it': it_mapping,
    'de': de_mapping,
    'es': es_mapping,
}

def sub_shift(json_data, mappings, *, lang, field = 'ingredients'):
    ingredients_key = f'{field}_{lang}'
    ents_key = f'ents_{lang}'

    for j, annotation in enumerate(json_data):
        text = annotation[ingredients_key]
        entities = annotation[ents_key]
        adjustment = 0
        pattern = re.compile('|'.join(re.escape(key) for key in mappings[lang].keys()))
        for match in re.finditer(pattern, text):
            match_index = match.start() + adjustment
            match_contents = match.group()
            contents = mappings[lang][match_contents]
            # if it's a list put it through the randomizer function
            if isinstance(contents, list):
                subbed_text = randomizer(contents)
            else:
                subbed_text = contents
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
        annotation[ingredients_key] = text
        # print(j, [annotation[ingredients_key][entities[i][0]:entities[i][1]] for i in range(len(entities))])
    return json_data

class Converter:
    def __init__(self, *, patterns) -> None:
        self.uom = None

    def str_frac_to_float(self, s):
        nums = [float(n) for n in s.split('/')]
        if len(nums)>1:
            return (nums[0]*self.uom)/nums[1]
        return nums[0]*self.uom

    @staticmethod
    def tighten_slash(s):
        slash_pattern = r'\s*/\s*'
        return re.sub(slash_pattern, '/', s.strip())

    def convert_num(self, input_string):
        if 'a' in input_string:
            self.sep = 'a'
            extremes = [s.replace('-', '') for s in input_string.split('a') if s != '']
        else:
            self.sep = '-'
            extremes = [s for s in input_string.split('-') if s != '']
        nums = []
        for ext in extremes:
            num_buffer = 0
            int_and_frac = self.tighten_slash(ext).split()
            num_buffer = sum(self.str_frac_to_float(part) for part in int_and_frac)
            if num_buffer>10:
                nums.append(int(num_buffer))
            else:
                nums.append(round(num_buffer, ndigits=1))
            
        return ' - '.join([str(n) for n in nums]) if self.sep == '-' else ' a '.join([str(n) for n in nums])
    
    def localize_ingredients(self, data, lang = 'it'):
        with open('./output.log', 'w') as f:
            for j, sample in enumerate(data['annotations']):
                ingredients_key = f'ingredients_{lang}'
                ents_key = f'ents_{lang}'
                text = sample[ingredients_key]
                ents = sample[ents_key]
                print(j, file = f)
                print(sample['ingredients_en'], file = f)
                print([sample['ingredients_en'][sample['ents_en'][i][0]:sample['ents_en'][i][1]] for i in range(len(sample['ents_en']))], file = f)
                print(text, file = f)
                print([text[ents[i][0]:ents[i][1]] for i in range(len(ents))], file = f)
                
                for pattern in pattern_list:
                    adjustment = 0
                    
                    regex = re.compile(pattern['pattern'], re.IGNORECASE) 
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
                sample[ingredients_key] = text
                print(text, file = f)
                print([text[ents[i][0]:ents[i][1]] for i in range(len(ents))], file = f)
                print('-----------------------------------------------------------', file = f)
        return data

pattern_list = [
    # fix uom
    # {'pattern': r'chili pepper(?!\w)', 'type': 'plain', 'sub': 'peperoncino'},
    # {'pattern': r'cancellato(?!\w)', 'type': 'plain', 'sub': 'strofinate'},
    # {'pattern': r'(?<=cucchiai )come(?!\w)', 'type': 'plain', 'sub': 'menta'},
    # {'pattern': r'(?<=ramoscello fresco )come(?!\w)', 'type': 'plain', 'sub': 'menta'},
    # {'pattern': r'come(?=\s(?:foglie|foglia))', 'type': 'plain', 'sub': 'menta'},

    {'pattern': r'(\d[-/\.\,\d\s]*)tazz.(?!\w)', 'ratio': 236, 'uom': 'g', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:(?:once fluide)|(?:oncia fluida)|once|oncia|oz\.|oz)(?!\w)', 'ratio': 28.35, 'uom': 'g', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\sa]*)(?:chili|chilo|lbs?\.?|libbre|pounds|pound|sterline|sterlina)(?!\w)', 'ratio': 0.45, 'uom': 'kg', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:di pollice|pollic.|\")(?!\w)', 'ratio': 2.54, 'uom': 'cm', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:pinte|pinta)(?!\w)', 'ratio': 473, 'uom': 'ml', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)(?:quartini|quartino)(?!\w)', 'ratio': 946, 'uom': 'ml', 'type': 'quantity'},
    {'pattern': r'(\d[-/\.\,\d\s]*)gallon(?!\w)', 'ratio': 3.78, 'uom': 'l', 'type': 'quantity'},

    {'pattern': r'tazz.(?!\w)', 'ratio': 236, 'uom': 'g', 'type': 'uom'},
    {'pattern': r'(?:(?:once fluide)|(?:oncia fluida)|once|oncia|oz\.|oz)(?!\w)', 'ratio': 28.35, 'uom': 'g', 'type': 'uom'},
    {'pattern': r'(?:chili|chilo|lbs?\.?|libbre|pounds|pound|sterline|sterlina)(?!\w)', 'ratio': 0.45, 'uom': 'kg', 'type': 'uom'},
    {'pattern': r'(?:di pollice|pollic.|\")(?!\w)', 'ratio': 2.54, 'uom': 'cm', 'type': 'uom'},
    {'pattern': r'(?:pinte|pinta)(?!\w)', 'ratio': 473, 'uom': 'ml', 'type': 'uom'},
    {'pattern': r'(?:quartini|quartino)(?!\w)', 'ratio': 946, 'uom': 'ml', 'type': 'uom'},
    {'pattern': r'gallon(?!\w)', 'ratio': 3.78, 'uom': 'l', 'type': 'uom'},

    # {'pattern': r'tigli(?!\w)', 'type': 'plain', 'sub': 'lime'},
    # {'pattern': r'orso(?!\w)', 'type': 'plain', 'sub': 'birra'},
    # {'pattern': r'orecchie(?!\w)', 'type': 'plain', 'sub': 'pannocchie'},
    # {'pattern': r'(?<=\d\s)misurini(?!\w)', 'type': 'plain', 'sub': 'palline'},
    # {'pattern': r'marrone(?= zucchero)', 'type': 'plain', 'sub': 'di canna'},
    # {'pattern': r'marrone(?= chiaro zucchero)', 'type': 'plain', 'sub': 'di canna'},
    # {'pattern': r'marrone(?= scuro zucchero)', 'type': 'plain', 'sub': 'di canna'},
    # {'pattern': r'abbondanza(?= sale kosher)', 'type': 'plain', 'sub': 'in abbondanza'},
    # {'pattern': r'kg(?= lime)', 'type': 'plain', 'sub': 'chili'},
    # {'pattern': r'(?<=mini )peperoni', 'type': 'plain', 'sub': 'peperoncini'},
    # {'pattern': r'(?<=aglio )(?:chiodo di garofano|chiodi di garofano)', 'type': 'plain', 'sub': 'spicchio'},
    # {'pattern': r'(?:chiodo di garofano|chiodi di garofano)(?= aglio)', 'type': 'plain', 'sub': 'spicchio'},
    # {'pattern': r'[Zz]uppa di cipolle alla francese', 'type': 'plain', 'sub': 'zuppa di cipolle francese'},
    # {'pattern': r'sale di cipolla', 'type': 'plain', 'sub': 'sale alla cipolla'},
    # {'pattern': r'livido', 'type': 'plain', 'sub': 'schiacciato'},
    # {'pattern': r'caldo(?= peperoncino)', 'type': 'plain', 'sub': 'piccante'},    
    # {'pattern': r'[Ff]arina di tutti i tipi', 'type': 'plain', 'sub': 'farina 00'},
    # {'pattern': r'(?<=cannella )bastone', 'type': 'plain', 'sub': 'bastoncino'},
    # {'pattern': r'bastone(?= cannella)', 'type': 'plain', 'sub': 'bastoncino'},
    # {'pattern': r'(?<=burro )bastone', 'type': 'plain', 'sub': 'panetto'},
    # {'pattern': r'bastone(?= burro)', 'type': 'plain', 'sub': 'panetto'},
    # {'pattern': r'(?<=pepe \()[Ff]resco', 'type': 'plain', 'sub': 'appena'},
    # {'pattern': r'(?<=alghe \()filo', 'type': 'plain', 'sub': 'arame'}, 
    # {'pattern': r'(?<=mazzo )verdi', 'type': 'plain', 'sub': 'verdura'},
    # {'pattern': r'scuro(?= gocce di cioccolato)', 'type': 'plain', 'sub': 'fondente'},
    # {'pattern': r'scuro(?= cioccolato)', 'type': 'plain', 'sub': 'fondente'},
    # {'pattern': r'(?<=scolato )sfogliato', 'type': 'plain', 'sub': 'a lamelle'},
    # {'pattern': r'cubo(?= bistecche)', 'type': 'plain', 'sub': 'girello'},
    # {'pattern': r'(?<!\w)[Mm]aggio', 'type': 'plain', 'sub': 'maionese'},
    # {'pattern': r'noce di cocco', 'type': 'plain', 'sub': 'cocco'},
    # {'pattern': r'(?<=(?:uova|uovo) )(bianchi)', 'type': 'plain', 'sub': 'albumi'},
    # {'pattern': r'carton(?!\w)', 'type': 'plain', 'sub': 'cartone'},
    # {'pattern': r'(?<=matzah )tavole', 'type': 'plain', 'sub': 'fogli'},
    # {'pattern': r'barattolo(?= cola)', 'type': 'plain', 'sub': 'lattina'},
    # {'pattern': r'per person(?!\w)', 'type': 'plain', 'sub': 'per persona'},
    # {'pattern': r'caldo(?= salsa)', 'type': 'plain', 'sub': 'piccante'},
    # {'pattern': r'(?<=wafer al cioccolato \()scuro', 'type': 'plain', 'sub': 'fondente'},
    # {'pattern': r'(?<=maiale )costole', 'type': 'plain', 'sub': 'costine'},
    # {'pattern': r'(?<=farina \()semplice', 'type': 'plain', 'sub': '00'},
    # {'pattern': r'(?<!\w)la miscela', 'type': 'plain', 'sub': 'il preparato'},
    # {'pattern': r'(?<=patata )pelli', 'type': 'plain', 'sub': 'bucce'},
    # {'pattern': r'bastoncini(?= burro)', 'type': 'plain', 'sub': 'panetti'},
    # {'pattern': r'bastoncini(?= salato burro)', 'type': 'plain', 'sub': 'panetti'},
    # {'pattern': r'(?<=circa 20 )bastoncini', 'type': 'plain', 'sub': 'punte'},
    # {'pattern': r'in polvere(?= zucchero)', 'type': 'plain', 'sub': 'a velo'},
    # {'pattern': r'barattolo(?= soda)', 'type': 'plain', 'sub': 'lattina'},
    # {'pattern': r'(?<=lime )soda', 'type': 'plain', 'sub': 'bibita'},
    # {'pattern': r'(?<=lattina )soda', 'type': 'plain', 'sub': 'bibita'},
    # {'pattern': r'barattolo(?= tonno)', 'type': 'plain', 'sub': 'lattina'},
    # {'pattern': r'(?<=2 kg )manzo in scatola punta di petto', 'type': 'plain', 'sub': 'punta di petto di manzo in scatola'},
    # {'pattern': r'(?<=4 fette )manzo in scatola', 'type': 'plain', 'sub': 'petto di manzo in scatola'},
    # {'pattern': r'(?<=0,9 - 1,4 kg )petto di manzo in scatola', 'type': 'plain', 'sub': 'punta di petto di manzo in scatola'},
    # {'pattern': r'(?<=vaniglia )fagiolo', 'type': 'plain', 'sub': 'baccello'},
    # {'pattern': r'(?<=denocciolato )data', 'type': 'plain', 'sub': 'dattero'},
    # {'pattern': r'(?<=arrostito rosso )pepe', 'type': 'plain', 'sub': 'peperone'},
    # {'pattern': r'pepe(?= \(a cubetti\))', 'type': 'plain', 'sub': 'peperone'},
    # {'pattern': r'pani(?!\w)', 'type': 'plain', 'sub': 'pagnotte'},
    
]

converter = Converter(patterns=pattern_list)

tgt_lang = 'es'
# json_file = f'/home/pgajo/food/data/TASTEset/data/EW-TASTE/EW-TT-MT_en-{tgt_lang}_context.json'
# json_file = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/SW-TASTE_en-it_DEEPL_unaligned.json'
json_file = '/home/pgajo/food/src/extract/mycolombianrecipes.json'

with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
# data = data['annotations']
# data = label_studio_to_tasteset(data, label_field=label_field)

# from icecream import ic
# ic(data)
# # clean up translation mistakes by doing simple substitutions first with this function
data = sub_shift(data, mappings, field = 'instructions', lang = 'en')
data = sub_shift(data, mappings, field = 'instructions', lang = tgt_lang)

with open(json_file[:-5] + '_fix_TS.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

# data = tasteset_to_label_studio(data['annotations'], label_field=label_field)
# localized_data = data
# # localized_data = converter.localize_ingredients(localized_data)
# # save to updated json file with different name
# with open(json_file[:-5] + '_fix.json', 'w', encoding='utf-8') as file:
#     json.dump(localized_data, file, ensure_ascii=False, indent=4)