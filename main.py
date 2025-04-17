from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import math
import re
from collections import Counter

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")


def calculate_tfidf(text: str):
    # Preprocess text: lowercase, remove punctuation, split into words
    words = re.findall(r'\b\w+\b', text.lower())

    # Calculate Term Frequency (TF)
    word_counts = Counter(words)
    total_words = len(words)
    tf = {word: count / total_words for word, count in word_counts.items()}

    # Calculate Inverse Document Frequency (IDF)
    # Since we have only one document, IDF will be simplified
    # In a multi-document scenario, IDF = log(N/df) where N is number of documents
    # Here, we'll use log(1/df) for single document
    idf = {word: math.log(1 / (count / total_words)) for word, count in word_counts.items()}

    # Calculate TF-IDF
    tfidf = {word: tf[word] * idf[word] for word in word_counts}

    # Prepare results
    results = [
        {"word": word, "tf": tf[word], "idf": idf[word], "tfidf": tfidf[word]}
        for word in word_counts
    ]

    # Sort by IDF in descending order and take top 50
    results = sorted(results, key=lambda x: x["idf"], reverse=True)[:50]

    return results


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    # Read and process file
    try:
        content = await file.read()
        text = content.decode('utf-8')

        # Calculate TF-IDF
        results = calculate_tfidf(text)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "results": results
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")