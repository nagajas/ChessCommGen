import requests
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm

links_file_path = "saved_files/saved_links.txt"

output_dir_path = "saved_files"
os.makedirs(output_dir_path, exist_ok=True)

output_file_path = os.path.join(output_dir_path, "commentary.txt")

all_data = 0

with open(links_file_path, "r") as file:
    links = [line.strip() for line in file.readlines()]

with open(output_file_path,'w') as f:
    for idx, link in enumerate(tqdm(links), start=1):
        try:
            r = requests.get(link)
            
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, 'html.parser')
                
                rows = soup.find_all('tr')
                
                for row in rows:
                    move_td = row.find('td', rowspan='2', style=lambda value: value and 'vertical-align: top' in value)
                    comment_td = row.find('td', style="vertical-align: top;")
                    
                    if move_td and comment_td:
                        move_text = move_td.get_text(separator=' ', strip=True)
                        move_text = ' '.join(move_text.split())
                        
                        comment_text = comment_td.get_text(separator=' ', strip=True)
                        comment_text = ' '.join(comment_text.split())

                        f.write(f"{move_text} <sep> {comment_text}\n")
                        all_data += 1
                
            else:
                print(f"Failed to fetch link {idx} with status code {r.status_code}")
        except Exception as e:
            print(f"Error while processing link {idx}: {e}")

print(f"Scraped data for {all_data} entries.")
print(f"All data saved in: {output_file_path}")
