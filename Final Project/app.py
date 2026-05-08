from flask import Flask, render_template, request

import pickle
import numpy as np
import re

from sklearn.metrics.pairwise import cosine_similarity

# create flask app
app = Flask(__name__)

# load saved files
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

df = pickle.load(open("captions_dataframe.pkl", "rb"))

# preprocessing function
def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', '', text)
    
    return text

# homepage
@app.route("/")

def home():
    
    sample_queries = [
        "angiographic image shows normal coronary artery",
        "CT scan demonstrating pulmonary embolism",
        "MRI scan showing brain tumor",
        "xray image of fractured bone"
    ]
    
    return render_template(
        "index.html",
        sample_queries=sample_queries
    )
    #return render_template("index.html")

# search function
@app.route("/search", methods=["POST"])

def search():
    
    query = request.form["query"]
    
    clean_query = clean_text(query)
    
    # tfidf vector
    query_vector = tfidf.transform([clean_query])
    
    # cosine similarity
    similarity = cosine_similarity(query_vector,
                                   tfidf_matrix)
    
    # top 5 results
    top_index = np.argsort(similarity[0])[::-1][:5]
    
    results = []
    
    for i in top_index:
        
        results.append({
            "image_id": df.iloc[i]["image_id"],
            "caption": df.iloc[i]["caption"],
            "score": round(similarity[0][i], 4)
        })
    
    return render_template("index.html",
                           results=results,
                           query=query)

# run app
if __name__ == "__main__":
    
    app.run(debug=True)