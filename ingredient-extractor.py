from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv

# Function to scrape a single page for its recipes and append to CSV
import re  # import regular expression library

def extract_ingredients_html(html_filename):
    print(f"Extracting ingredients from: {html_filename}")
    with open(html_filename, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # First, look for the gz-list-ingredients section
    ingredient_list = soup.find('dl', {'class': 'gz-list-ingredients'})
    
    iq_list = []

    if ingredient_list:
        ingredients = ingredient_list.find_all('dd', {'class': 'gz-ingredient'})
        
        for ing in ingredients:
            ingredient = ing.find('a').text if ing.find('a') else ''
            quantity = ing.find('span').text if ing.find('span') else ''
            
            iq_list.append(ingredient)
            # in 'quantity' replace any \t with a space
            quantity = ' '.join(quantity.replace('\t', ' ').split())
            iq_list.append(quantity)
    else:
        # If gz-list-ingredients not found, search all instances of gz-content-recipe for ul-based ingredients
        ingredient_lists_div = soup.find_all('div', {'class': 'gz-content-recipe'})

        for ingredient_list_div in ingredient_lists_div:
            ingredient_list_ul = ingredient_list_div.find('ul')
            
            if ingredient_list_ul:
                ingredients = ingredient_list_ul.find_all('li', dir='ltr')
                
                for ing in ingredients:
                    ing_text = ing.find('span').text if ing.find('span') else ''
                    
                    # Using regular expression to find quantity and unit of measure
                    pattern = re.compile(r'(\d+[\d.,]*)\s*([a-zA-Z]{1,3}\b)(?<!\bdi\b)')
                    match = pattern.search(ing_text)
                    
                    if match:
                        quantity = match.group(0)  # Full match
                        # Removing quantity from ingredient text
                        ingredient = ing_text.replace(quantity, '').strip()
                    else:
                        quantity = ''
                        ingredient = ing_text.strip()
                    
                    iq_list.append(quantity)
                    iq_list.append(ingredient)
                    
    # Padding lists to have uniform size
    iq_list += [''] * (50 - len(iq_list))
    iq_list += [html_filename]

    # Appending to DataFrame
    df_row = pd.DataFrame([iq_list])

    # Append row to CSV
    df_row.to_csv('ingredients.csv', mode='a', header=False, index=False)


# Prepare the CSV with column names
columns = []
for i in range(1, 51):
    columns += [f'i_{i}']+[f"q_{i}"]
columns += ['url']

with open('ingredients.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)

# iterate through the directory of saved pages C:\Users\paolo\repos\food\scraped-pages-giallozafferano
import os
for filename in os.listdir(r"C:\Users\paolo\repos\food\scraped-pages-giallozafferano"):
    if filename.endswith(".html"):
        extract_ingredients_html(fr"C:\Users\paolo\repos\food\scraped-pages-giallozafferano\{filename}")
        continue
    else:
        continue