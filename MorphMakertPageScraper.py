from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

# Base URL of the website (we'll change the page number in the loop)
base_url = 'https://morphmarket.com/us/c/reptiles/lizards/crested-geckos?page='

# Initialize the Chrome driver
driver = webdriver.Chrome()  # Or webdriver.Firefox() if using Firefox

# Iterate through all 700 pages
for page_num in range(1, 303):  # Loop from page 1 to page 700
    url = f"{base_url}{page_num}"
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    # Parse the fully loaded HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # List to hold data for the current page
    animal_data = []

    # Find all animal cards on the page
    animal_cards = soup.find_all('a', class_='animalCard--avL0R')

    # Iterate over each animal card and extract information
    for card in animal_cards:
        # Get the name
        name_tag = card.find('h4', class_='animalTitle--lz_Ps')
        name = name_tag.text if name_tag else None

        # Get the image link
        img_tag = card.find('picture').find('img')
        img_link = img_tag['src'] if img_tag else None

        # Get the profile link
        profile_link = card['href'] if card else None

        # Append data to the list
        animal_data.append({
            "Animal Name": name,
            "Image Link": img_link,
            "Profile Link": profile_link
        })

    # Convert the data for the current page into a DataFrame and append it to the CSV file
    df = pd.DataFrame(animal_data)
    df.to_csv('animal_data.csv', mode='a', index=False, header=(page_num == 1))

    print(f"Data from page {page_num} saved to animal_data.csv")

# Close the driver after scraping all pages
driver.quit()
print("All pages processed and data saved.")
