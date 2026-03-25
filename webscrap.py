from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()
SECRAPER_API_KEY = os.getenv("SECRAPER_API_KEY")
def ws(name):
    params = {
    "engine": "google_shopping",
    "q": name,
    "api_key": "79eb4be8c9c1974cc734d847f92b6459c0aece75ae48158b83b0585c21f9583b",
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results["shopping_results"]
    return(shopping_results[0:5])
