from bs4 import BeautifulSoup
import os
import urllib.request
from icecream import ic
from tqdm.auto import tqdm
import re

url = "https://www.mycolombianrecipes.com/recipes/"
f = urllib.request.urlopen(url)
raw_mainpage_html = f.read()
html_soup_object = BeautifulSoup(raw_mainpage_html, 'html.parser')
categories = html_soup_object.find_all(class_="feast-category-index-list feast-grid-half feast-desktop-grid-fourth feast-image-round")

# categories = categories[:1]

df_list = []
category_page_url_list = []
for i, el in tqdm(enumerate(categories), total=len(categories)):
    category_url_list = [category['href'] for category in el.find_all('a', href = True)]
    # category_url_list = category_url_list[:1]
    for category_url in tqdm(category_url_list, total=len(category_url_list), desc='Gathering all page URLs'):
        # ic(category_url)
        f = urllib.request.urlopen(category_url)
        raw_category_page_html = f.read()
        category_page_object = BeautifulSoup(raw_category_page_html, 'html.parser')
        content_section = category_page_object.find_all(class_="content")[0]
        page_list = content_section.find_all('div', class_='archive-pagination pagination')[0].find_all('a')
        # ic(page_list[-2])
        max_page = int(re.search(re.compile(r'page/(\d)'), page_list[-2]['href']).group(1))
        if max_page is None:
            max_page = 1
        
        for i in range(max_page):
            if i == 0:
                category_page_url_list.append(category_url)
            else:
                category_page_url_list.append(category_url+f'page/{i+1}/')

# ic(category_page_url_list)
# category_page_url_list = category_page_url_list[:1]

for category_page_url in category_page_url_list:
    f = urllib.request.urlopen(category_page_url)
    raw_category_page_html = f.read()
    category_page_object = BeautifulSoup(raw_category_page_html, 'html.parser')
    content_section = category_page_object.find_all(class_="content")[0]

    recipes = content_section.find_all('article')

    for recipe_en in tqdm(recipes, total=len(recipes), desc=f'Category URL: {category_page_url}'):
        recipe_url_en = [r['href'] for r in recipe_en.find_all('a', class_ = 'entry-image-link', href = True)][0]
        ic(recipe_url_en)
        recipe_entry = {}
        f = urllib.request.urlopen(recipe_url_en)
        page_html_en = f.read()
        recipe_page_object_en = BeautifulSoup(page_html_en, 'html.parser')
        page_urls_en = recipe_page_object_en.find_all('a', href = True)
        for page_url_en in page_urls_en:
            if page_url_en.text == 'Español':
                recipe_url_es = page_url_en['href']

        recipe_entry['url_en'] = recipe_url_en
        recipe_entry['url_es'] = recipe_url_es

        f = urllib.request.urlopen(recipe_url_es)
        page_html_es = f.read()
        recipe_page_object_es = BeautifulSoup(page_html_es, 'html.parser')

        ingredient_list_en = recipe_page_object_en.find_all('div', class_ = 'wprm-recipe-ingredient-group')
        if ingredient_list_en:
            ingredients_en = ' ; '.join([ing.text for ing in ingredient_list_en[0].find_all('li', class_ = 'wprm-recipe-ingredient')])
        else:
            ingredients_en = 'no_ingredients_en'
        
        ingredient_list_es = recipe_page_object_es.find_all('div', class_ = 'wprm-recipe-ingredient-group')
        if ingredient_list_es:
            ingredients_es = ' ; '.join([ing.text for ing in ingredient_list_es[0].find_all('li', class_ = 'wprm-recipe-ingredient')])
        else:
            ingredients_es = 'no_ingredients_es'
            
        recipe_entry['ingredients_en'] = ingredients_en
        recipe_entry['ingredients_es'] = ingredients_es

        instructions_list_en = recipe_page_object_en.find_all('div', class_ = 'wprm-recipe-instruction-group')
        if instructions_list_en:
            instructions_en = '\n'.join([ing.text for ing in instructions_list_en[0].find_all('li', class_ = 'wprm-recipe-instruction')])
        else:
            instructions_en = 'no_instructions_en'

        instructions_list_es = recipe_page_object_es.find_all('div', class_ = 'wprm-recipe-instruction-group')
        if instructions_list_es:
            instructions_es = '\n'.join([ing.text for ing in instructions_list_es[0].find_all('li', class_ = 'wprm-recipe-instruction')])
        else:
            instructions_es = 'no_instructions_es'

        recipe_entry['instructions_en'] = instructions_en
        recipe_entry['instructions_es'] = instructions_es

        nutrition_list = recipe_page_object_en.find_all('div', class_ = 'wprm-nutrition-label-container wprm-nutrition-label-container-grouped wprm-block-text-normal')
        if nutrition_list:
            nutrition = '\n'.join(set([ing.text for ing in nutrition_list[0].find_all('span')]))
        else:
            nutrition = 'no_nutrition'

        recipe_entry['nutrition'] = nutrition

        df_list.append(recipe_entry)

import pandas as pd
# ic(df_list)
df = pd.DataFrame(df_list)
# print(df)

parent_dir = '/home/pgajo/food/src/extract'
jsonl_filename = 'mycolombianrecipes.jsonl'
df.to_json(os.path.join(parent_dir, jsonl_filename), orient='records', lines = True)
json_filename = 'mycolombianrecipes.json'
df.to_json(os.path.join(parent_dir, json_filename), orient='records')