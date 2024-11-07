import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Load the Excel file
excel_file_path = 'test_scrape_list.xlsx'
df = pd.read_excel(excel_file_path)

# Directory to save individual CSV files for each scraped URL
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Configure Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Function to scrape content from a URL
def scrape_content(url):
    try:
        driver.get(url)
        time.sleep(2)  # Allow time for page to load
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        return text_content
    except Exception as e:
        print(f"Failed to retrieve content from {url}: {e}")
        return None

# Function to save a DataFrame as a CSV file
def save_dataframe(df, filename):
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=False)

# Iterate over each row in the DataFrame and scrape the content from each link
for index, row in df.iterrows():
    url = row['link']
    filename = row['filename']
    csv_path = os.path.join(output_dir, f"{filename}.csv")

    # Check if the file already exists
    if os.path.exists(csv_path):
        print(f"Row {index + 1}: File already exists, skipping.")
        continue

    content = scrape_content(url)

    if content:
        article_data = {
            'Date': row['Email Summary Date'],
            'Article': row['Article'],
            'Summary': row['Summary'],
            'Source': row['Source'],
            'Tag': row['Topic/Issue Tag'],
            'link': url,
            'Content': content
        }
        article_df = pd.DataFrame([article_data])
        save_dataframe(article_df, filename)
        print(f"Row {index + 1}: Successfully scraped and saved content from {url}.")
    else:
        print(f"Row {index + 1}: Failed to scrape content from {url}.")

driver.quit()
print("Scraping complete. CSV files saved.")
