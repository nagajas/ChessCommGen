import requests
from bs4 import BeautifulSoup
import pickle
import os
from tqdm import tqdm

root_url = "https://gameknot.com"

saved_links = []

os.makedirs("saved_files", exist_ok=True)

text_file_path = "saved_files/saved_links.txt"

for page_index in tqdm(range(290), desc="Processing pages"):
    page_url = f"https://gameknot.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p={page_index}"

    r = requests.get(page_url)
    
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        
        for elem in soup.find_all('tr', class_=["evn_list", "odd_list"]):
            list_of_links = elem.find_all('a')
            
            if len(list_of_links) > 1:
                href = list_of_links[1].get('href')
                if href:
                    full_link = root_url + href
                    saved_links.append(full_link)
        
        #print(f"Processed page {page_index}")
    else:
        #print(f"Failed to fetch page {page_index} with status code {r.status_code}")
        break

pickle_file_path = "saved_files/saved_links.p"
with open(pickle_file_path, "wb") as pickle_file:
    pickle.dump(saved_links, pickle_file)

with open(text_file_path, "w") as text_file:
    for link in saved_links:
        text_file.write(link + "\n")

print(f"Saved {len(saved_links)} links:")
print(f"- Pickle file: {pickle_file_path}")
print(f"- Text file: {text_file_path}")
