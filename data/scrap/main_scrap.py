import requests
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm
import numpy as np

links_file_path = "saved_files/saved_links.p"
page_count_file_path = "saved_files/extra_pages.p"

output_dir_path = "./saved_files"
os.makedirs(output_dir_path, exist_ok=True)

output_file_path = os.path.join(output_dir_path, "move_commentary.txt")

all_data = []

saved_links = np.load(links_file_path, allow_pickle=True)
extra_pages = np.load(page_count_file_path, allow_pickle=True)

with open(output_file_path,'w') as f:
    for idx, link in enumerate(tqdm(saved_links)):
        for p in range(extra_pages[idx]):
            try:
                r = requests.get(link+f"&pg={p}")
                
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
                           
                else:
                    print(f"Failed to fetch link {idx} with status code {r.status_code}")
                f.write('\n') 
            except Exception as e:
                print(f"Error while processing link {idx}: {e}")

print(f"Scraped data for {len(all_data)} entries.")
print(f"All data saved in: {output_file_path}")

with open(output_file_path, 'r') as f:
    data = f.readlines()

new_output_path = os.path.join(output_dir_path, "game_commentary.txt")

with open(new_output_path, 'w') as f:
    newgames = 0
    for line in data:
        moves_list, _ = line.split("<sep>")
        if moves_list.startswith("1.") and not moves_list.startswith("1..."):
            newgames += 1
            f.write('\n\n')
        
        f.write(line.strip()+'<move>')

print(f"Total games found: {newgames}")
print(f"Data saved in: {new_output_path}")