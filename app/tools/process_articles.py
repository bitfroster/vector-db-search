import json
import wikipediaapi
import re
from config import config


def download_wikipedia_articles(titles):
    print("Downloading of articles started...")
    try:
        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent='UApp/1.0 (uzma51@embrille.com)')
        json_data = {'text': []}
        output_file = config.get('input_file', '/app/data/articles.json')

        with open(output_file, 'w', encoding='utf-8') as json_file:
            for title in titles:
                page = wiki_wiki.page(title)
                json_data['text'].append(clean_text(page.text))

            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Downloading failed. Details: {e}")

    print("Downloading of articles finished.")


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text
