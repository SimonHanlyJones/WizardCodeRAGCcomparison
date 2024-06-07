# pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer

#######################################
# Fetch all URLS from the python index.
#######################################

# def get_all_urls(url):
#     """Fetch all URLs from a given webpage."""
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check for HTTP errors

#         soup = BeautifulSoup(response.text, 'html.parser')
#         links = soup.find_all('a')  # Find all anchor tags

#         urls = []
#         for link in links:
#             href = link.get('href')
#             if href:
#                 full_url = urljoin(url, href)  # Create absolute URL
#                 urls.append(full_url)

#         return urls
#     except requests.RequestException as e:
#         print(f"Error fetching {url}: {e}")
#         return []

# #######################################
# # Fetch all text from a URL.
# #######################################

# def get_page_text(url):
#     """Fetch and clean the text from a given URL."""
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an error for bad status codes
#         soup = BeautifulSoup(response.text, 'html.parser')
#         text = ' '.join(soup.stripped_strings)  # Extract text and remove extra whitespace
#         return text
#     except requests.RequestException as e:
#         print(f"Error fetching {url}: {e}")
#         return None

# #######################################
# # Fetch all text from a list of URLs.
# #######################################

# def main(urls):
#     """Main function to fetch text from a list of URLs."""
#     texts = []
#     for url in urls:
#         text = get_page_text(url)
#         if text:
#             texts.append(text)
#     return texts

# #######################################
# # URL Helpers
# #######################################

# def remove_from_hash(input_string):
#     # Find the index of the first occurrence of '#'
#     hash_index = input_string.find('#')

#     if hash_index != -1:
#         return input_string[:hash_index]
#     else:
#         return input_string

# def remove_duplicates(input_list):
#     seen = set()
#     unique_list = []
#     for item in input_list:
#         if item not in seen:
#             unique_list.append(item)
#             seen.add(item)
#     return unique_list

# def filter_strings_by_prefix(input_list, prefix):
#     # Use list comprehension to filter out strings that don't start with the prefix
#     filtered_list = [s for s in input_list if s.startswith(prefix)]
#     return filtered_list

# def remove_strings_with_prefix(input_list, prefix):
#     # Use list comprehension to filter out strings that start with the prefix
#     filtered_list = [s for s in input_list if not s.startswith(prefix)]
#     return filtered_list

# #######################################
# # Get Python Docs
# #######################################

# webpage_url = "https://python.readthedocs.io/en/latest/genindex-all.html"
# urls = get_all_urls(webpage_url)

# # remove internal anchors from urls
# urls = [remove_from_hash(url) for url in urls]
# urls = remove_duplicates(urls)
# urls = filter_strings_by_prefix(urls,'https://python.readthedocs.io/en/latest/library')

# docs = main(urls)

import json
import os
import lancedb
import pandas as pd

# Read the list from a file
with open('pychunks.json', 'r') as file:
    docs = json.load(file)

print("Number of docs:", len(docs))


docs = [x.replace('#', '-') for x in docs]

# Now we need to embed these documents and put them into a "vector store" or
# "vector db" that we will use for semantic search and retrieval.

# Embeddings setup
name="all-MiniLM-L12-v2"
model = SentenceTransformer(name)
 
def embed_batch(batch):
    return [model.encode(sentence) for sentence in batch]
 
def embed(sentence):
    return model.encode(sentence)
 
# LanceDB setup
os.mkdir(".lancedb" + name)
uri = ".lancedb" + name
db = lancedb.connect(uri)
 
# Create a dataframe with the chunk ids and chunks
metadata = []
for i in range(len(docs)):
    metadata.append([
        i,
        docs[i]
    ])
doc_df = pd.DataFrame(metadata, columns=["chunk", "text"])
 
# Embed the documents
data = with_embeddings(embed_batch, doc_df)
 
# Create the DB table and add the records.
db.create_table("linux", data=data)
table = db.open_table("linux")
table.add(data=data)

# usage example
# uri = ".lancedb" + name
# db = lancedb.connect(uri)
# table = db.open_table("linux")
