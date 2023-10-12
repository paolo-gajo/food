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
    button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "button.amecp_button-accetto.iubenda-cs-accept-btn"))
    )
    button.click()
except Exception as e:
    print(f"Could not click the button: {e}")

# Function to scrape a single page for its recipes and append to CSV
def scrape_page(page_url):
    driver.get(page_url)
    time.sleep(2)  # Give time for the page to load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Look for the gz-list-ingredients section
    ingredient_list = soup.find('dl', {'class': 'gz-list-ingredients'})
    
    if ingredient_list:
        iq_list = []
        
        ingredients = ingredient_list.find_all('dd', {'class': 'gz-ingredient'})
        
        for ing in ingredients:
            ingredient = ing.find('a').text if ing.find('a') else ''
            quantity = ing.find('span').text if ing.find('span') else ''
            
            iq_list.append(ingredient)
            # in 'quantity' replace any \t with a space
            quantity = ' '.join(quantity.replace('\t', ' ').split())
            iq_list.append(quantity)

        
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

# TODO: Write code to iterate through all recipe pages on the website
# For demonstration, we are only scraping the first page
scrape_page(start_url)

# Close the WebDriver
driver.close()