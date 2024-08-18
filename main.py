from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from pymongo import MongoClient
import os

# Download 'punkt' for NLTK tokenization
nltk.download('punkt')

# Set up stemmer and TF-IDF vectorizer
stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)

def preprocess(text):
    return ' '.join(tokenize_and_stem(text))

# Load and preprocess data (Lazy Loading)
def load_and_preprocess_data():
    try:
        # Load only the necessary columns and preprocess in chunks if file is large
        data = pd.read_csv('finafideprod.csv', usecols=['Product', 'Maincategory', 'Subcategory', 'Producttype', 'Brand'])
    except FileNotFoundError:
        raise FileNotFoundError("The file 'finafideprod.csv' was not found.")

    data['processed_text'] = data.apply(
        lambda row: preprocess(f"{row['Product']} {row['Maincategory']} {row['Subcategory']} {row['Producttype']} {row['Brand']}"),
        axis=1
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])
    return data, tfidf_matrix

# Perform fuzzy search with reduced memory footprint
def fuzzy_search(query, choices):
    return process.extractOne(query, choices)

# Perform keyword search
def substring_search(query, text):
    return query.lower() in text.lower()

# Perform the search operation
def search_products(query):
    data, tfidf_matrix = load_and_preprocess_data()
    query_processed = preprocess(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    data['similarity'] = cosine_similarities

    # Use only the maximum fuzzy score to reduce operations
    def fuzzy_scores(row):
        scores = [
            fuzzy_search(query, [row['Product']])[1],
            fuzzy_search(query, [row['Maincategory']])[1],
            fuzzy_search(query, [row['Subcategory']])[1],
            fuzzy_search(query, [row['Producttype']])[1],
            fuzzy_search(query, [row['Brand']])[1]
        ]
        return max(scores)

    data['fuzzy_score'] = data.apply(fuzzy_scores, axis=1)

    # Keyword-based search
    def keyword_search(row):
        fields = [row['Product'], row['Maincategory'], row['Subcategory'], row['Producttype'], row['Brand']]
        return sum(substring_search(query, field) for field in fields)

    data['keyword_score'] = data.apply(keyword_search, axis=1)

    # Combine scores: cosine similarity, fuzzy score, and keyword score
    data['combined_score'] = (
        0.5 * data['similarity'] +
        0.3 * (data['fuzzy_score'] / 100.0) +
        0.2 * data['keyword_score']
    )

    # Filter results
    threshold = 0.1
    filtered_results = data[
        (data['combined_score'] >= threshold) &
        (data['keyword_score'] > 0)
    ]

    filtered_results = filtered_results.sort_values(by='combined_score', ascending=False).head(5)

    if filtered_results.empty:
        return "No products found. Please try a different query."

    return filtered_results[['Product', 'Maincategory', 'Subcategory', 'Producttype', 'Brand']].to_dict(orient='records')

# Connect to MongoDB Atlas
mongo_uri = os.getenv('MONGO_URI', 'your_default_mongo_uri')
client = MongoClient(mongo_uri)
db = client['mydatabase']
collection = db['search_queries']

def log_search_query(query):
    search_entry = {
        'query': query,
        'timestamp': pd.Timestamp.now()
    }
    try:
        collection.insert_one(search_entry)
    except Exception as e:
        print(f"Error logging search query: {e}")

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

@app.post("/search/")
async def perform_search(request: SearchRequest):
    query = request.query
    results = search_products(query)
    if results == "No products found. Please try a different query.":
        return {"message": results}
    else:
        log_search_query(query)
        return {"results": results}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
