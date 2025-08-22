from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fuzzywuzzy import fuzz, process
import pandas as pd
import os
import re
import unicodedata
from typing import List, Dict, Optional
import pyarabic.araby as araby
import arabic_reshaper
from bidi.algorithm import get_display
import jellyfish
from rapidfuzz import fuzz as rapidfuzz_fuzz, process as rapidfuzz_process

app = FastAPI(
    title="Names Detection API",
    description="Professional Arabic name similarity detection system with advanced text processing",
    version="2.0.0"
)

# Arabic text processing utilities
class ArabicTextProcessor:
    """Advanced Arabic text processing for name similarity detection"""
    
    @staticmethod
    def normalize_arabic_text(text: str) -> str:
        """Comprehensive Arabic text normalization"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove diacritics (tashkeel)
        text = araby.strip_diacritics(text)
        
        # Normalize different forms of alef
        text = araby.normalize_alef(text)
        
        # Normalize different forms of teh marbuta and heh
        text = araby.normalize_teh(text)
        
        # Remove tatweel (kashida) - elongation character
        text = araby.strip_tatweel(text)
        
        # Remove non-Arabic characters except spaces
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]', '', text)
        
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    @staticmethod
    def get_arabic_tokens(text: str) -> List[str]:
        """Extract meaningful Arabic tokens from text"""
        normalized = ArabicTextProcessor.normalize_arabic_text(text)
        tokens = normalized.split()
        
        # Filter out very short tokens (less than 2 characters)
        meaningful_tokens = [token for token in tokens if len(token) >= 2]
        
        return meaningful_tokens
    
    @staticmethod
    def get_phonetic_code(text: str) -> str:
        """Generate phonetic code for Arabic text"""
        # Normalize first
        normalized = ArabicTextProcessor.normalize_arabic_text(text)
        
        # Simple Arabic phonetic mapping for similar sounds
        phonetic_map = {
            'ث': 'س', 'ذ': 'ز', 'ص': 'س', 'ض': 'د', 'ط': 'ت', 'ظ': 'ز',
            'ئ': 'ء', 'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ة': 'ه'
        }
        
        for arabic_char, replacement in phonetic_map.items():
            normalized = normalized.replace(arabic_char, replacement)
        
        return normalized

class SimilarityCalculator:
    """Advanced similarity calculation for Arabic names"""
    
    def __init__(self):
        self.weights = {
            'exact_match': 1.0,
            'normalized_match': 0.98,
            'token_set': 0.90,
            'token_sort': 0.85,
            'partial': 0.70,
            'phonetic': 0.75,
            'ratio': 0.60
        }
    
    def calculate_comprehensive_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """Calculate multiple similarity scores for Arabic names"""
        
        # Normalize both names
        norm1 = ArabicTextProcessor.normalize_arabic_text(name1)
        norm2 = ArabicTextProcessor.normalize_arabic_text(name2)
        
        # Get tokens
        tokens1 = ArabicTextProcessor.get_arabic_tokens(name1)
        tokens2 = ArabicTextProcessor.get_arabic_tokens(name2)
        
        # Get phonetic codes
        phone1 = ArabicTextProcessor.get_phonetic_code(name1)
        phone2 = ArabicTextProcessor.get_phonetic_code(name2)
        
        scores = {}
        
        # Exact match
        scores['exact'] = 100.0 if name1 == name2 else 0.0
        
        # Normalized match
        scores['normalized'] = 100.0 if norm1 == norm2 else 0.0
        
        # Traditional fuzzy matching on normalized text
        scores['ratio'] = rapidfuzz_fuzz.ratio(norm1, norm2)
        scores['partial_ratio'] = rapidfuzz_fuzz.partial_ratio(norm1, norm2)
        scores['token_sort_ratio'] = rapidfuzz_fuzz.token_sort_ratio(norm1, norm2)
        scores['token_set_ratio'] = rapidfuzz_fuzz.token_set_ratio(norm1, norm2)
        
        # Phonetic similarity
        scores['phonetic'] = rapidfuzz_fuzz.ratio(phone1, phone2)
        
        # Token-based similarity
        if tokens1 and tokens2:
            token_scores = []
            for token1 in tokens1:
                best_token_score = max([rapidfuzz_fuzz.ratio(token1, token2) for token2 in tokens2], default=0)
                token_scores.append(best_token_score)
            scores['token_avg'] = sum(token_scores) / len(token_scores) if token_scores else 0
        else:
            scores['token_avg'] = 0
        
        return scores
    
    def get_weighted_similarity(self, scores: Dict[str, float]) -> float:
        """Calculate weighted final similarity score"""
        
        # Use the best scores with appropriate weights
        weighted_score = 0
        total_weight = 0
        
        # Prioritize exact and normalized matches
        if scores.get('exact', 0) > 0:
            return scores['exact']
        
        if scores.get('normalized', 0) > 0:
            return scores['normalized'] * self.weights['normalized_match']
        
        # Calculate weighted average of other scores
        score_weights = [
            (scores.get('token_set_ratio', 0), self.weights['token_set']),
            (scores.get('token_sort_ratio', 0), self.weights['token_sort']),
            (scores.get('partial_ratio', 0), self.weights['partial']),
            (scores.get('phonetic', 0), self.weights['phonetic']),
            (scores.get('ratio', 0), self.weights['ratio']),
            (scores.get('token_avg', 0), 0.75)  # Token average weight
        ]
        
        for score, weight in score_weights:
            if score > 0:
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = max(scores.values()) if scores else 0
        
        return round(final_score, 2)

# Configuration
class Config:
    MIN_SIMILARITY_THRESHOLD = 65
    SUSPICIOUS_THRESHOLD = 75
    MAX_RESULTS = 10
    ENABLE_PHONETIC_MATCHING = True
    ENABLE_TOKEN_MATCHING = True

# Initialize components
arabic_processor = ArabicTextProcessor()
similarity_calculator = SimilarityCalculator()

# Load blacklist from CSV
def load_blacklist():
    """Load and normalize blacklist names from CSV file"""
    csv_path = os.path.join(os.path.dirname(__file__), "blacklist.csv")
    try:
        df = pd.read_csv(csv_path)
        names = df['name'].tolist()
        
        # Normalize all blacklist names for better matching
        normalized_names = []
        for name in names:
            if name and isinstance(name, str):
                normalized = arabic_processor.normalize_arabic_text(name)
                if normalized:  # Only add non-empty normalized names
                    normalized_names.append(name)  # Keep original name for display
        
        print(f"Loaded {len(normalized_names)} names from blacklist")
        return normalized_names
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
    similarity_details: Optional[Dict[str, float]] = None
    normalized_input: Optional[str] = None
    normalized_blacklist: Optional[str] = None

class NameCheckResponse(BaseModel):
    input_name: str
    normalized_input: str
    is_suspicious: bool
    max_similarity: float
    confidence_level: str
    matches: List[SimilarityResult]
    processing_info: Dict[str, str] = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Professional Arabic Names Detection API",
        "description": "Advanced Arabic name similarity detection with comprehensive text processing",
        "version": "2.0.0",
        "features": [
            "Arabic text normalization and diacritics handling",
            "Phonetic similarity matching",
            "Token-based analysis",
            "Weighted similarity scoring",
            "Multiple similarity algorithms",
            "Confidence level assessment"
        ],
        "endpoints": {
            "POST /check-name": "Comprehensive name analysis with detailed scoring",
            "POST /check-name-simple": "Quick name check with enhanced processing",
            "GET /blacklist": "Get the current blacklist",
            "GET /config": "Get current API configuration",
            "POST /config": "Update API configuration",
            "GET /health": "Health check and system status"
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

@app.get("/config")
async def get_config():
    """Get current API configuration"""
    return {
        "min_similarity_threshold": Config.MIN_SIMILARITY_THRESHOLD,
        "suspicious_threshold": Config.SUSPICIOUS_THRESHOLD,
        "max_results": Config.MAX_RESULTS,
        "enable_phonetic_matching": Config.ENABLE_PHONETIC_MATCHING,
        "enable_token_matching": Config.ENABLE_TOKEN_MATCHING,
        "similarity_weights": similarity_calculator.weights
    }

@app.post("/config")
async def update_config(config_update: dict):
    """Update API configuration (for fine-tuning)"""
    updated_fields = []
    
    if "min_similarity_threshold" in config_update:
        Config.MIN_SIMILARITY_THRESHOLD = max(0, min(100, config_update["min_similarity_threshold"]))
        updated_fields.append("min_similarity_threshold")
    
    if "suspicious_threshold" in config_update:
        Config.SUSPICIOUS_THRESHOLD = max(0, min(100, config_update["suspicious_threshold"]))
        updated_fields.append("suspicious_threshold")
    
    if "max_results" in config_update:
        Config.MAX_RESULTS = max(1, min(50, config_update["max_results"]))
        updated_fields.append("max_results")
    
    return {
        "message": "Configuration updated successfully",
        "updated_fields": updated_fields,
        "current_config": {
            "min_similarity_threshold": Config.MIN_SIMILARITY_THRESHOLD,
            "suspicious_threshold": Config.SUSPICIOUS_THRESHOLD,
            "max_results": Config.MAX_RESULTS
        }
    }

@app.post("/check-name", response_model=NameCheckResponse)
async def check_name(request: NameCheckRequest):
    """
    Professional Arabic name similarity detection with advanced text processing
    Returns comprehensive similarity analysis with detailed scoring
    """
    if not BLACKLIST:
        raise HTTPException(status_code=500, detail="Blacklist not loaded")
    
    input_name = request.name.strip()
    if not input_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    # Normalize input name
    normalized_input = arabic_processor.normalize_arabic_text(input_name)
    if not normalized_input:
        raise HTTPException(status_code=400, detail="Invalid Arabic name provided")
    
    # Calculate similarities with each blacklisted name
    similarities = []
    processing_stats = {
        "total_comparisons": str(len(BLACKLIST)),
        "above_threshold": "0",
        "processing_method": "advanced_arabic_nlp"
    }
    
    for blacklisted_name in BLACKLIST:
        # Calculate comprehensive similarity
        similarity_scores = similarity_calculator.calculate_comprehensive_similarity(
            normalized_input, blacklisted_name
        )
        
        # Get weighted final score
        final_score = similarity_calculator.get_weighted_similarity(similarity_scores)
        
        if final_score >= Config.MIN_SIMILARITY_THRESHOLD:
            current_above = int(processing_stats["above_threshold"])
            processing_stats["above_threshold"] = str(current_above + 1)
            
            similarities.append(SimilarityResult(
                blacklisted_name=blacklisted_name,
                similarity_percentage=final_score,
                similarity_details=similarity_scores,
                normalized_input=normalized_input,
                normalized_blacklist=arabic_processor.normalize_arabic_text(blacklisted_name)
            ))
    
    # Sort by similarity percentage (descending)
    similarities.sort(key=lambda x: x.similarity_percentage, reverse=True)
    
    # Determine suspicion level and confidence
    max_sim = similarities[0].similarity_percentage if similarities else 0
    is_suspicious = max_sim >= Config.SUSPICIOUS_THRESHOLD
    
    # Determine confidence level
    if max_sim >= 95:
        confidence = "very_high"
    elif max_sim >= 85:
        confidence = "high"
    elif max_sim >= 70:
        confidence = "medium"
    elif max_sim >= 60:
        confidence = "low"
    else:
        confidence = "very_low"
    
    return NameCheckResponse(
        input_name=input_name,
        normalized_input=normalized_input,
        is_suspicious=is_suspicious,
        max_similarity=max_sim,
        confidence_level=confidence,
        matches=similarities[:Config.MAX_RESULTS],
        processing_info=processing_stats
    )

@app.post("/check-name-simple")
async def check_name_simple(request: NameCheckRequest):
    """
    Simple endpoint with enhanced Arabic processing that returns the best match
    """
    if not BLACKLIST:
        raise HTTPException(status_code=500, detail="Blacklist not loaded")
    
    input_name = request.name.strip()
    if not input_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    # Normalize input
    normalized_input = arabic_processor.normalize_arabic_text(input_name)
    
    # Find best match using advanced similarity
    best_score = 0
    best_match = None
    best_details = {}
    
    for blacklisted_name in BLACKLIST:
        similarity_scores = similarity_calculator.calculate_comprehensive_similarity(
            normalized_input, blacklisted_name
        )
        final_score = similarity_calculator.get_weighted_similarity(similarity_scores)
        
        if final_score > best_score:
            best_score = final_score
            best_match = blacklisted_name
            best_details = similarity_scores
    
    return {
        "input_name": input_name,
        "normalized_input": normalized_input,
        "matched_name": best_match,
        "similarity_percentage": round(best_score, 2),
        "is_suspicious": best_score >= Config.SUSPICIOUS_THRESHOLD,
        "confidence_level": "high" if best_score >= 85 else "medium" if best_score >= 70 else "low",
        "similarity_details": best_details
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)