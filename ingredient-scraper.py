from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time

# Initialize WebDriver
driver = webdriver.Chrome(executable_path=r"C:\Users\paolo\OneDrive\Desktop\chromedriver-win64\chromedriver.exe")

# DataFrame to store ingredients and quantities
df = pd.DataFrame(columns=[f'i_{i}' for i in range(1, 51)] + [f'q_{i}' for i in range(1, 51)])

# Function to scrape a single page for its recipes
def scrape_page(page_url):
    global df
    driver.get(page_url)
    time.sleep(2)  # Give time for the page to load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Look for the gz-list-ingredients section
    ingredient_list = soup.find('dl', {'class': 'gz-list-ingredients'})
    
    if ingredient_list:
        i_list = []
        q_list = []
        
        ingredients = ingredient_list.find_all('dd', {'class': 'gz-ingredient'})
        
        for ing in ingredients:
            ingredient = ing.find('a').text if ing.find('a') else ''
            quantity = ing.find('span').text if ing.find('span') else ''
            
            i_list.append(ingredient)
            q_list.append(quantity)
            
        # Padding lists to have uniform size
        i_list += [''] * (50 - len(i_list))
        q_list += [''] * (50 - len(q_list))
        
        # Appending to DataFrame
        df = df.append(pd.Series(i_list + q_list, index=df.columns), ignore_index=True)

# First URL to start scraping from
start_url = "https://ricette.giallozafferano.it/"

# TODO: Write code to iterate through all recipe pages on the website
# For demonstration, we are only scraping the first page
scrape_page(start_url)

# Save to CSV
df.to_csv('/home/pgajo/working/food/giallo-zafferano-corpus/ingredients.csv', index=False
