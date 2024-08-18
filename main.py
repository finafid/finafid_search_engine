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
import nltk

# # Create a custom directory for nltk_data
# nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# # Ensure the directory exists
# if not os.path.exists(nltk_data_dir):
#     os.makedirs(nltk_data_dir)

# Set the path for NLTK data
# nltk.data.path.append(nltk_data_dir)

# Download 'punkt' to the custom directory
nltk.download('punkt')
nltk.download('punkt_tab')


# Define tokenizer and stemmer for TF-IDF and fuzzy matching
stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)

def preprocess(text):
    return ' '.join(tokenize_and_stem(text))

# Load and preprocess data
try:
    data = pd.read_csv('finafideprod.csv')  # Updated CSV file name
except FileNotFoundError:
    raise FileNotFoundError("The file 'finafideprod.csv' was not found.")

required_columns = ['Product', 'Maincategory', 'Subcategory', 'Producttype', 'Brand']  # Updated data fields
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing one or more required columns in the DataFrame")

data['processed_text'] = data.apply(
    lambda row: preprocess(row['Product'] + ' ' + row['Maincategory'] + ' ' + row['Subcategory'] + ' ' + row['Producttype'] + ' ' + row['Brand']), axis=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])

def fuzzy_search(query, choices):
    result = process.extractOne(query, choices)
    return result if result else (None, 0)

def substring_search(query, text):
    return query.lower() in text.lower()

def search_products(query):
    query_processed = preprocess(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    data['similarity'] = cosine_similarities

    # Fuzzy match across Brand, Producttype, Subcategory, Maincategory, and Product
    def fuzzy_scores(product, maincategory, subcategory, producttype, brand):
        scores = {
            'Product': fuzzy_search(query, [product])[1],
            'Maincategory': fuzzy_search(query, [maincategory])[1],
            'Subcategory': fuzzy_search(query, [subcategory])[1],
            'Producttype': fuzzy_search(query, [producttype])[1],
            'Brand': fuzzy_search(query, [brand])[1]
        }
        return max(scores.values())

    data['fuzzy_score'] = data.apply(lambda row: fuzzy_scores(row['Product'], row['Maincategory'], row['Subcategory'], row['Producttype'], row['Brand']),
                                      axis=1)

    # Keyword-based search (check for the presence of query terms in the text fields)
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

    # Filter results where the combined score is greater than a threshold
    threshold = 0.1  # Lowered threshold to ensure more results are considered
    filtered_results = data[
        (data['combined_score'] >= threshold) &
        (data.apply(lambda row: any(substring_search(query, field) for field in [row['Product'], row['Maincategory'], row['Subcategory'], row['Producttype'], row['Brand']]), axis=1))
    ]

    filtered_results = filtered_results.sort_values(by='combined_score', ascending=False).head(5)

    if filtered_results.empty:
        return "No products found. Please try a different query."

    return filtered_results[['Product', 'Maincategory', 'Subcategory', 'Producttype', 'Brand']].to_dict(orient='records')

# Connect to MongoDB Atlas
mongo_uri = os.getenv('MONGO_URI', 'mongodb+srv://Deepfinafid:H9sat87XxjLPMCYA@deepfinafid.ywcsxq8.mongodb.net/')
client = MongoClient(mongo_uri)
db = client['mydatabase']  # Replace 'mydatabase' with your database name
collection = db['search_queries']  # Replace 'search_queries' with your collection name

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
    port = int(os.getenv("PORT", 8000))  # Use the port from the environment variable or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
