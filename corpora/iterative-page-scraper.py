from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv

# import undetected_chromedriver as uc
# driver = uc.Chrome(headless=True,use_subprocess=False)

# Initialize WebDriver
driver = webdriver.Chrome()

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
    print('I clicked!')
except Exception as e:
    print(f"Could not click the button: {e}")

# Function to scrape a single page for its recipes and append to CSV
import re  # import regular expression library

def scrape_page(page_url):
    print(f"Scraping page: {page_url}")
    driver.get(page_url)
    time.sleep(2)  # Give time for the page to load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # save soup content to a file with the page_url.split('/')[-1] as the filename
    print(page_url.split('/')[-1][:-1])
    with open(fr"C:\Users\paolo\repos\food\scraped-pages-giallozafferano\{page_url.split('/')[-1][:-1]}.html", "w", encoding="utf-8") as file:
        file.write(str(soup))

# First URL to start scraping from
start_url = "https://ricette.giallozafferano.it"

# Iterate through N pages from list.txt
with open('url_list_test.txt', 'r') as f:
    for url in f:
        scrape_page(url)

# Close the WebDriver
driver.close()
