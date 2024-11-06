from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import time
import re

# Percorso al WebDriver
webdriver_path = "C:\\Users\\Fabio\\Desktop\\WEBDRIVER\\chromedriver-win64\\chromedriver.exe"

# Configura il servizio WebDriver
service = Service(webdriver_path)
driver = webdriver.Chrome(service=service)

# Percorso file di output
output_file_path = "C:\\Users\\Fabio\\Desktop\\BERT with WEBDRIVER\\reviews.txt"

# Funzione per rimuovere "Full Review", "Original Score", e date
def clean_review_text(review_text):
    # Rimuove righe con "Full Review", "Original Score", e date come "Feb 7, 2024"
    cleaned_text = re.sub(r'(Full Review.*?(\||$))|(Original Score:.*?(\||$))|(\|\s*\w+\s\d{1,2},\s\d{4})|(\b\w+\s\d{1,2},\s\d{4}\b)', '', review_text, flags=re.IGNORECASE)
    return cleaned_text.strip()

try:
    # Apri il sito web
    url = "https://www.rottentomatoes.com/m/past_lives/reviews"
    driver.get(url)

    # Chiudi il banner dei cookie, se presente
    try:
        reject_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Reject All')]")
        reject_button.click()
        print("Banner dei cookie chiuso con 'Reject All'.")
        time.sleep(1)
    except NoSuchElementException:
        print("Pulsante 'Reject All' non trovato, procedo...")

    # Apri il file in modalità scrittura
    with open(output_file_path, "w", encoding="utf-8") as file:
        while True:
            # Cerca e stampa le recensioni attuali
            try:
                reviews_container = driver.find_element(By.CLASS_NAME, "reviews-container")
                review_table = reviews_container.find_element(By.CLASS_NAME, "review_table")
                review_items = review_table.find_elements(By.CSS_SELECTOR, '[data-qa="review-item"]')

                for review_item in review_items:
                    try:
                        review_text = review_item.find_element(By.CLASS_NAME, "review-text-container")
                        raw_content = review_text.text
                        cleaned_content = clean_review_text(raw_content)
                        if cleaned_content:  # Salva solo se il contenuto non è vuoto
                            # Scrive la recensione nel formato richiesto
                            file.write(f'"{cleaned_content}",\n\n')
                    except NoSuchElementException:
                        print("Classe 'review-text-container' non trovata in questo review-item.")
            
            except NoSuchElementException:
                print("Errore nel trovare le recensioni.")

            # Trova e clicca sul pulsante "Load More"
            try:
                load_more_button = driver.find_element(By.CSS_SELECTOR, '[data-loadmoremanager="btnLoadMore:click"]')
                driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                time.sleep(1)
                load_more_button.click()
                print("Pulsante 'Load More' cliccato con successo.")
                time.sleep(2)
            except NoSuchElementException:
                print("Pulsante 'Load More' non trovato. Fine dei contenuti.")
                break
            except ElementClickInterceptedException:
                print("Un elemento blocca il pulsante. Ritenta.")
                time.sleep(2)

finally:
    # Chiudi il browser
    driver.quit()
    print(f"Recensioni salvate in: {output_file_path}")
