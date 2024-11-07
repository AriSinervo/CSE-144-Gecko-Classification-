import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os

# Load URLs from CSV file
csv_file = 'animal_data.csv'  # Replace with the path to your CSV file
url_data = pd.read_csv(csv_file, header=None)

# Extract URLs after the second comma (assuming each line has data in this format)
urls = url_data[2]  # Adjust this index based on your CSV structure

# Set up Selenium WebDriver (assuming ChromeDriver is in PATH)
driver = webdriver.Chrome()  # Or webdriver.Firefox() if using Firefox

for url in urls:
    if url == "Profile Link":
        continue
    temp = "https://morphmarket.com" + url.strip()
    driver.get(temp)  # Ensure any whitespace is removed

    # Wait for the page to fully load (adjust time as necessary)
    time.sleep(1)

    # Parse the fully loaded HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the animal's ID
    name_tag = soup.find_all('div', class_='labelValueContainer--z1CP3')
    nameID = None
    for name in name_tag:
        if "Animal ID" in str(name):
            n = str(name)
            start = int(n.index("</b><span>")) + 10
            end = int(n.index("</span>"))
            nameID = n[start:end]
            break  # Exit loop once Animal ID is found

    # Extract traits
    tag_link = soup.find('div', class_="MuiBox-root css-0")
    tags = str(tag_link)
    listoftraits = []
    #print(tags)
    while "njoQS" in tags or "YbPYK" in tags:
        #print(tags)
        tags=tags[tags.index("</span>")-25:]
        listoftraits.append(tags[tags.index('">')+2: tags.index("</span>")])
        tags = tags[tags.index("</span>") + 10:]

    # Check if data already exists in CSV


    # Prepare the new data entry
    new_entry = {
        "Animal Id": nameID,
        "Tag": listoftraits,
        "Source URL": url  # Track the URL source for each animal
    }

    # Append the new entry to the CSV
    df = pd.DataFrame([new_entry])  # Create a DataFrame for this single entry
    df.to_csv('animal_profile.csv', mode='a', header=not os.path.exists('animal_profile.csv'), index=False)

    print(f"Data for Animal ID {nameID} saved.")

# Close the driver after all pages are scraped
driver.quit()

print("Data scraping complete.")
