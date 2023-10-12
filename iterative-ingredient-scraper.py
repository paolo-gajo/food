from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv


# Initialize WebDriver
driver = webdriver.Chrome(r"C:\Users\paolo\repos\food\chromedriver.exe")

# Navigate to the starting page
driver.get("https://ricette.giallozafferano.it")

# Wait for the button to appear and click it
try:
    # Wait for up to 10 seconds before throwing a TimeoutException
    button = WebDriverWait(driver, 10).until(
        # Condition: An element with the CSS selector matching the accept button is present
        EC.presence_of_element_located((By.CSS_SELECTOR, ".amecp_button-accetto.iubenda-cs-accept-btn"))
    )
    button.click()
except Exception as e:
    print(f"Could not click the button: {e}")

# Function to scrape a single page for its recipes and append to CSV
import re  # import regular expression library

def scrape_page(page_url):
    print(f"Scraping page: {page_url}")
    driver.get(page_url)
    time.sleep(2)  # Give time for the page to load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
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
                    
                    # Split ingredient and quantity based on the presence of a digit before the first space
                    first_space_idx = ing_text.find(' ')
                    if first_space_idx != -1 and re.search(r'\d', ing_text[:first_space_idx]):
                        ingredient, quantity = ing_text.split(' ', 1)
                    else:
                        ingredient, quantity = ing_text, ''
                    
                    iq_list.append(quantity.strip())
                    iq_list.append(ingredient.strip())
                    
    # Padding lists to have uniform size
    iq_list += [''] * (50 - len(iq_list))

    # Appending to DataFrame
    df_row = pd.DataFrame([iq_list])

    # Append row to CSV
    df_row.to_csv('ingredients.csv', mode='a', header=False, index=False)


# Prepare the CSV with column names
columns = []
for i in range(1, 51):
    columns += [f'i_{i}']+[f"q_{i}"]
with open('ingredients.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)

# First URL to start scraping from
start_url = "https://ricette.giallozafferano.it"

# Iterate through N pages from list.txt
with open('url_list_test.txt', 'r') as f:
    for url in f:
        scrape_page(url)

# Close the WebDriver
driver.close()
