from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fuzzywuzzy import fuzz, process
import pandas as pd
import os
from typing import List, Dict

app = FastAPI(
    title="Names Detection API",
    description="API for checking name similarity against a blacklist",
    version="1.0.0"
)

# Load blacklist from CSV
def load_blacklist():
    """Load blacklist names from CSV file"""
    csv_path = os.path.join(os.path.dirname(__file__), "blacklist.csv")
    try:
        df = pd.read_csv(csv_path)
        return df['name'].tolist()
    except Exception as e:
        print(f"Error loading blacklist: {e}")
        return []

# Global blacklist
BLACKLIST = load_blacklist()

class NameCheckRequest(BaseModel):
    name: str

class SimilarityResult(BaseModel):
    blacklisted_name: str
    similarity_percentage: float

class NameCheckResponse(BaseModel):
    input_name: str
    is_suspicious: bool
    max_similarity: float
    matches: List[SimilarityResult]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Names Detection API",
        "description": "Check name similarity against blacklist",
        "endpoints": {
            "POST /check-name": "Check a name against the blacklist",
            "GET /blacklist": "Get the current blacklist",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "blacklist_loaded": len(BLACKLIST) > 0,
        "blacklist_count": len(BLACKLIST)
    }

@app.get("/blacklist")
async def get_blacklist():
    """Get the current blacklist"""
    return {
        "blacklist": BLACKLIST,
        "count": len(BLACKLIST)
    }

@app.post("/check-name", response_model=NameCheckResponse)
async def check_name(request: NameCheckRequest):
    """
    Check if a name is similar to any name in the blacklist
    Returns similarity percentages for all matches above threshold
    """
    if not BLACKLIST:
        raise HTTPException(status_code=500, detail="Blacklist not loaded")
    
    input_name = request.name.strip()
    if not input_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    # Calculate similarity with each blacklisted name
    similarities = []
    threshold = 60  # Minimum similarity percentage to consider
    
    for blacklisted_name in BLACKLIST:
        # Use different similarity algorithms
        ratio = fuzz.ratio(input_name, blacklisted_name)
        partial_ratio = fuzz.partial_ratio(input_name, blacklisted_name)
        token_sort_ratio = fuzz.token_sort_ratio(input_name, blacklisted_name)
        token_set_ratio = fuzz.token_set_ratio(input_name, blacklisted_name)
        
        # Take the maximum similarity score
        max_similarity = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        if max_similarity >= threshold:
            similarities.append(SimilarityResult(
                blacklisted_name=blacklisted_name,
                similarity_percentage=round(max_similarity, 2)
            ))
    
    # Sort by similarity percentage (descending)
    similarities.sort(key=lambda x: x.similarity_percentage, reverse=True)
    
    # Determine if suspicious (similarity >= 70%)
    max_sim = similarities[0].similarity_percentage if similarities else 0
    is_suspicious = max_sim >= 70
    
    return NameCheckResponse(
        input_name=input_name,
        is_suspicious=is_suspicious,
        max_similarity=max_sim,
        matches=similarities[:10]  # Return top 10 matches
    )

@app.post("/check-name-simple")
async def check_name_simple(request: NameCheckRequest):
    """
    Simple endpoint that returns just the highest similarity match
    """
    if not BLACKLIST:
        raise HTTPException(status_code=500, detail="Blacklist not loaded")
    
    input_name = request.name.strip()
    if not input_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    # Find the best match
    best_match = process.extractOne(input_name, BLACKLIST, scorer=fuzz.token_set_ratio)
    
    if best_match:
        similarity = best_match[1]
        matched_name = best_match[0]
        
        return {
            "input_name": input_name,
            "matched_name": matched_name,
            "similarity_percentage": similarity,
            "is_suspicious": similarity >= 70
        }
    else:
        return {
            "input_name": input_name,
            "matched_name": None,
            "similarity_percentage": 0,
            "is_suspicious": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)