from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
from fuzzywuzzy import fuzz, process
import pandas as pd
import os
import re
import unicodedata
from typing import List, Dict, Optional, Union, Any
import pyarabic.araby as araby
import arabic_reshaper
from bidi.algorithm import get_display
import jellyfish
from rapidfuzz import fuzz as rapidfuzz_fuzz, process as rapidfuzz_process
from functools import lru_cache
import hashlib
import logging
import time
import asyncio
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state for advanced features
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Arabic Names Detection API...")
    yield
    logger.info("Shutting down Arabic Names Detection API...")

app = FastAPI(
    title="Advanced Arabic Names Detection API",
    description="Enterprise-grade Arabic name similarity detection with comprehensive security and performance features",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Configuration
class Config:
    MIN_SIMILARITY_THRESHOLD = 65
    SUSPICIOUS_THRESHOLD = 75
    MAX_RESULTS = 10
    ENABLE_PHONETIC_MATCHING = True
    ENABLE_TOKEN_MATCHING = True
    
    # Security settings
    MAX_INPUT_LENGTH = 200
    MIN_INPUT_LENGTH = 2
    REQUEST_RATE_LIMIT = 100  # requests per minute
    MAX_BATCH_SIZE = 50
    
    # Performance settings
    CACHE_SIZE = 10000
    MAX_CONCURRENT_REQUESTS = 100
    
    # Monitoring
    ENABLE_DETAILED_LOGGING = True
    ENABLE_PERFORMANCE_TRACKING = True

# Enhanced Arabic text processing utilities
class ArabicTextProcessor:
    """Advanced Arabic text processing for name similarity detection with comprehensive language support"""
    
    # Arabic-to-Latin transliteration mapping
    TRANSLITERATION_MAP = {
        'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
        'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
        'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
        'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
        'ء': 'a', 'أ': 'a', 'إ': 'i', 'آ': 'aa', 'ة': 'h', 'ى': 'a'
    }
    
    # Latin-to-Arabic reverse mapping (for handling transliterated input)
    REVERSE_TRANSLITERATION = {v: k for k, v in TRANSLITERATION_MAP.items()}
    
    # Common Arabic name patterns and titles
    ARABIC_TITLES = {'أبو', 'ابو', 'أم', 'ام', 'الدكتور', 'دكتور', 'الأستاذ', 'استاذ', 'شيخ', 'الشيخ'}
    ARABIC_CONNECTORS = {'بن', 'ابن', 'ال', 'آل', 'عبد', 'عبدالله', 'عبدالرحمن'}
    
    @staticmethod
    @lru_cache(maxsize=Config.CACHE_SIZE)
    def normalize_arabic_text(text: str) -> str:
        """Enhanced Arabic text normalization with transliteration support"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle mixed Arabic-Latin text
        text = ArabicTextProcessor._handle_mixed_script(text)
        
        # Remove diacritics (tashkeel)
        text = araby.strip_diacritics(text)
        
        # Normalize different forms of alef
        text = araby.normalize_alef(text)
        
        # Normalize different forms of teh marbuta and heh
        text = araby.normalize_teh(text)
        
        # Remove tatweel (kashida) - elongation character
        text = araby.strip_tatweel(text)
        
        # Enhanced Arabic character normalization
        text = ArabicTextProcessor._normalize_arabic_variants(text)
        
        # Remove non-Arabic characters except spaces and common punctuation
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-\.]', '', text)
        
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    @staticmethod
    def _handle_mixed_script(text: str) -> str:
        """Handle text with both Arabic and Latin characters"""
        # If text contains both Arabic and Latin, try to convert Latin to Arabic
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        
        if has_arabic and has_latin:
            # Convert common Latin transliterations to Arabic
            for latin, arabic in ArabicTextProcessor.REVERSE_TRANSLITERATION.items():
                if len(latin) > 1:  # Prioritize longer matches
                    text = re.sub(rf'\b{re.escape(latin)}\b', arabic, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _normalize_arabic_variants(text: str) -> str:
        """Normalize various Arabic character variants"""
        # Additional character normalizations
        normalizations = {
            'ي': 'ى',  # Normalize different forms of yeh
            'ك': 'ک',  # Normalize different forms of kaf
            'ة': 'ه',  # Normalize teh marbuta to heh for better matching
        }
        
        for original, replacement in normalizations.items():
            text = text.replace(original, replacement)
        
        return text
    
    @staticmethod
    @lru_cache(maxsize=Config.CACHE_SIZE // 2)
    def get_arabic_tokens(text: str) -> tuple:
        """Enhanced token extraction with title and connector handling"""
        normalized = ArabicTextProcessor.normalize_arabic_text(text)
        tokens = normalized.split()
        
        # Filter out titles and connectors for better matching
        meaningful_tokens = []
        for token in tokens:
            if (len(token) >= 2 and 
                token not in ArabicTextProcessor.ARABIC_TITLES and
                token not in ArabicTextProcessor.ARABIC_CONNECTORS):
                meaningful_tokens.append(token)
        
        return tuple(meaningful_tokens)
    
    @staticmethod
    @lru_cache(maxsize=Config.CACHE_SIZE // 2)
    def get_phonetic_code(text: str) -> str:
        """Enhanced phonetic code generation with regional variations"""
        # Normalize first
        normalized = ArabicTextProcessor.normalize_arabic_text(text)
        
        # Enhanced Arabic phonetic mapping for similar sounds
        phonetic_map = {
            'ث': 'س', 'ذ': 'ز', 'ص': 'س', 'ض': 'د', 'ط': 'ت', 'ظ': 'ز',
            'ئ': 'ء', 'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ة': 'ه',
            'ك': 'ق', 'ق': 'ك',  # Common pronunciation variations
            'ج': 'غ',  # Regional dialect variations
            'ع': 'ا',  # Vowel-like sounds
        }
        
        for arabic_char, replacement in phonetic_map.items():
            normalized = normalized.replace(arabic_char, replacement)
        
        return normalized
    
    @staticmethod
    def get_transliteration(text: str) -> str:
        """Convert Arabic text to Latin transliteration"""
        result = ""
        for char in text:
            if char in ArabicTextProcessor.TRANSLITERATION_MAP:
                result += ArabicTextProcessor.TRANSLITERATION_MAP[char]
            elif char.isspace():
                result += " "
        return result
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect primary language of the input text"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = arabic_chars + latin_chars
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        if arabic_ratio > 0.7:
            return "arabic"
        elif arabic_ratio < 0.3:
            return "latin"
        else:
            return "mixed"

class AdvancedSimilarityCalculator:
    """Enhanced similarity calculation with context-aware algorithms and false positive reduction"""
    
    def __init__(self):
        self.weights = {
            'exact_match': 1.0,
            'normalized_match': 0.98,
            'token_set': 0.90,
            'token_sort': 0.85,
            'partial': 0.70,
            'phonetic': 0.75,
            'ratio': 0.60,
            'transliteration': 0.80,
            'semantic': 0.85
        }
        # Enhanced cache with TTL
        self._similarity_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 3600  # 1 hour
    
    def _get_cache_key(self, name1: str, name2: str) -> str:
        """Generate cache key for similarity calculation"""
        combined = f"{name1}|{name2}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[cache_key] < self._cache_ttl
    
    def calculate_comprehensive_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """Enhanced similarity calculation with context awareness"""
        
        # Check cache first
        cache_key = self._get_cache_key(name1, name2)
        if cache_key in self._similarity_cache and self._is_cache_valid(cache_key):
            return self._similarity_cache[cache_key]
        
        # Normalize both names
        norm1 = ArabicTextProcessor.normalize_arabic_text(name1)
        norm2 = ArabicTextProcessor.normalize_arabic_text(name2)
        
        # Get tokens
        tokens1 = ArabicTextProcessor.get_arabic_tokens(name1)
        tokens2 = ArabicTextProcessor.get_arabic_tokens(name2)
        
        # Get phonetic codes
        phone1 = ArabicTextProcessor.get_phonetic_code(name1)
        phone2 = ArabicTextProcessor.get_phonetic_code(name2)
        
        # Get transliterations for cross-language matching
        trans1 = ArabicTextProcessor.get_transliteration(norm1)
        trans2 = ArabicTextProcessor.get_transliteration(norm2)
        
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
        
        # Transliteration similarity (for cross-language matching)
        scores['transliteration'] = rapidfuzz_fuzz.ratio(trans1, trans2)
        
        # Enhanced token-based similarity
        scores['token_avg'] = self._calculate_token_similarity(tokens1, tokens2)
        
        # Context-aware semantic similarity
        scores['semantic'] = self._calculate_semantic_similarity(tokens1, tokens2)
        
        # Length-based penalty for significant size differences
        scores['length_penalty'] = self._calculate_length_penalty(norm1, norm2)
        
        # Cache the result with timestamp
        self._similarity_cache[cache_key] = scores
        self._cache_timestamps[cache_key] = time.time()
        
        # Manage cache size
        self._cleanup_cache()
        
        return scores
    
    def _calculate_token_similarity(self, tokens1: tuple, tokens2: tuple) -> float:
        """Enhanced token-based similarity calculation"""
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate bidirectional token similarity
        forward_scores = []
        for token1 in tokens1:
            best_score = max([rapidfuzz_fuzz.ratio(token1, token2) for token2 in tokens2], default=0)
            forward_scores.append(best_score)
        
        backward_scores = []
        for token2 in tokens2:
            best_score = max([rapidfuzz_fuzz.ratio(token1, token2) for token1 in tokens1], default=0)
            backward_scores.append(best_score)
        
        # Use the average of both directions for more balanced scoring
        avg_forward = sum(forward_scores) / len(forward_scores) if forward_scores else 0
        avg_backward = sum(backward_scores) / len(backward_scores) if backward_scores else 0
        
        return (avg_forward + avg_backward) / 2
    
    def _calculate_semantic_similarity(self, tokens1: tuple, tokens2: tuple) -> float:
        """Calculate semantic similarity using Arabic name patterns"""
        if not tokens1 or not tokens2:
            return 0.0
        
        # Check for common Arabic name patterns
        common_patterns = {
            'محمد': ['احمد', 'حمد'],
            'عبدالله': ['عبدالرحمن', 'عبدالعزيز'],
            'حسن': ['حسين', 'حسام'],
            'علي': ['عالي', 'عليان'],
        }
        
        semantic_score = 0.0
        matches = 0
        
        for token1 in tokens1:
            for token2 in tokens2:
                if token1 == token2:
                    semantic_score += 100.0
                    matches += 1
                elif token1 in common_patterns and token2 in common_patterns[token1]:
                    semantic_score += 85.0
                    matches += 1
                elif token2 in common_patterns and token1 in common_patterns[token2]:
                    semantic_score += 85.0
                    matches += 1
        
        if matches == 0:
            return 0.0
        
        return semantic_score / matches
    
    def _calculate_length_penalty(self, name1: str, name2: str) -> float:
        """Calculate penalty for significant length differences"""
        len1, len2 = len(name1), len(name2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        ratio = min(len1, len2) / max(len1, len2)
        return ratio * 100  # Convert to percentage
    
    def get_weighted_similarity(self, scores: Dict[str, float]) -> float:
        """Enhanced weighted similarity calculation with context awareness"""
        
        # Prioritize exact and normalized matches
        if scores.get('exact', 0) > 0:
            return scores['exact']
        
        if scores.get('normalized', 0) > 0:
            return scores['normalized'] * self.weights['normalized_match']
        
        # Apply length penalty
        length_penalty = scores.get('length_penalty', 100) / 100
        
        # Calculate weighted average with penalty
        weighted_score = 0
        total_weight = 0
        
        score_weights = [
            (scores.get('token_set_ratio', 0), self.weights['token_set']),
            (scores.get('token_sort_ratio', 0), self.weights['token_sort']),
            (scores.get('partial_ratio', 0), self.weights['partial']),
            (scores.get('phonetic', 0), self.weights['phonetic']),
            (scores.get('ratio', 0), self.weights['ratio']),
            (scores.get('token_avg', 0), 0.75),
            (scores.get('transliteration', 0), self.weights['transliteration']),
            (scores.get('semantic', 0), self.weights['semantic'])
        ]
        
        for score, weight in score_weights:
            if score > 0:
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = (weighted_score / total_weight) * length_penalty
        else:
            final_score = max(scores.values()) * length_penalty if scores else 0
        
        return round(final_score, 2)
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        if len(self._similarity_cache) > Config.CACHE_SIZE:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._cache_timestamps.items()
                if current_time - timestamp > self._cache_ttl
            ]
            
            for key in expired_keys:
                self._similarity_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
            
            # If still too large, remove oldest entries
            if len(self._similarity_cache) > Config.CACHE_SIZE:
                oldest_keys = sorted(
                    self._cache_timestamps.items(),
                    key=lambda x: x[1]
                )[:len(self._similarity_cache) - Config.CACHE_SIZE]
                
                for key, _ in oldest_keys:
                    self._similarity_cache.pop(key, None)
                    self._cache_timestamps.pop(key, None)

class SecurityValidator:
    """Enhanced security validation for all inputs"""
    
    @staticmethod
    def validate_input(text: str) -> tuple[bool, str]:
        """Comprehensive input validation"""
        if not text:
            return False, "Input cannot be empty"
        
        if len(text) > Config.MAX_INPUT_LENGTH:
            return False, f"Input too long (max {Config.MAX_INPUT_LENGTH} characters)"
        
        if len(text.strip()) < Config.MIN_INPUT_LENGTH:
            return False, f"Input too short (min {Config.MIN_INPUT_LENGTH} characters)"
        
        # Check for potential injection attempts
        suspicious_patterns = [
            r'<script.*?>', r'javascript:', r'onclick', r'onerror',
            r'union.*select', r'drop.*table', r'insert.*into',
            r'delete.*from', r'update.*set', r'exec.*sp_'
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, "Potentially malicious input detected"
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s\u0600-\u06FF]', text)) / len(text)
        if special_char_ratio > 0.3:
            return False, "Too many special characters"
        
        return True, "Valid"
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize input while preserving Arabic characters"""
        # Remove potential XSS characters
        text = re.sub(r'[<>"\']', '', text)
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        return text.strip()

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str, limit: int = Config.REQUEST_RATE_LIMIT) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        with self.lock:
            # Clean old requests
            client_requests = self.requests[client_id]
            while client_requests and client_requests[0] < minute_ago:
                client_requests.popleft()
            
            # Check limit
            if len(client_requests) >= limit:
                return False
            
            # Add current request
            client_requests.append(now)
            return True

# Enhanced Performance monitoring with detailed analytics
class AdvancedPerformanceMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset_stats()
        self.request_history = deque(maxlen=1000)  # Store last 1000 requests
        self.suspicious_patterns = defaultdict(int)
    
    def reset_stats(self):
        with self.lock:
            self.total_requests = 0
            self.cache_hits = 0
            self.total_comparisons = 0
            self.avg_processing_time = 0
            self.false_positives = 0
            self.true_positives = 0
            self.error_count = 0
            self.blocked_requests = 0
    
    def record_request(self, request_data: dict, comparisons: int = 0, cache_hit: bool = False):
        with self.lock:
            self.total_requests += 1
            self.total_comparisons += comparisons
            if cache_hit:
                self.cache_hits += 1
            
            self.request_history.append({
                'timestamp': datetime.now(),
                'data': request_data
            })
    
    def record_suspicious_pattern(self, pattern: str):
        with self.lock:
            self.suspicious_patterns[pattern] += 1
    
    def get_detailed_stats(self):
        with self.lock:
            recent_requests = len([r for r in self.request_history 
                                 if r['timestamp'] > datetime.now() - timedelta(hours=1)])
            
            cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
            avg_comparisons = self.total_comparisons / self.total_requests if self.total_requests > 0 else 0
            
            return {
                "total_requests": self.total_requests,
                "cache_hit_rate": round(cache_hit_rate, 2),
                "average_comparisons_per_request": round(avg_comparisons, 2),
                "total_comparisons": self.total_comparisons,
                "false_positives": self.false_positives,
                "true_positives": self.true_positives,
                "error_count": self.error_count,
                "blocked_requests": self.blocked_requests,
                "recent_requests_1h": recent_requests,
                "suspicious_patterns": dict(self.suspicious_patterns),
                "uptime": time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
    
    def start_monitoring(self):
        self._start_time = time.time()

class DynamicBlacklistManager:
    """Advanced blacklist management with runtime updates and versioning"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.blacklist = []
        self.normalized_blacklist = {}  # original -> normalized mapping
        self.version = 1
        self.last_modified = 0
        self.lock = threading.Lock()
        self.load_blacklist()
    
    def load_blacklist(self):
        """Load blacklist with error handling and validation"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning(f"Blacklist file not found: {self.csv_path}")
                return
            
            df = pd.read_csv(self.csv_path)
            new_blacklist = []
            new_normalized = {}
            
            for name in df['name'].tolist():
                if name and isinstance(name, str) and len(name.strip()) > 0:
                    cleaned_name = name.strip()
                    normalized = arabic_processor.normalize_arabic_text(cleaned_name)
                    
                    if normalized and len(normalized) >= 2:  # Minimum length check
                        new_blacklist.append(cleaned_name)
                        new_normalized[cleaned_name] = normalized
            
            with self.lock:
                self.blacklist = new_blacklist
                self.normalized_blacklist = new_normalized
                self.version += 1
                self.last_modified = time.time()
            
            logger.info(f"Loaded {len(new_blacklist)} names from blacklist (version {self.version})")
            
        except Exception as e:
            logger.error(f"Error loading blacklist: {e}")
    
    def reload_if_changed(self):
        """Reload blacklist if file has been modified"""
        try:
            if os.path.exists(self.csv_path):
                file_mtime = os.path.getmtime(self.csv_path)
                if file_mtime > self.last_modified:
                    logger.info("Blacklist file modified, reloading...")
                    self.load_blacklist()
        except Exception as e:
            logger.error(f"Error checking blacklist modification: {e}")
    
    def add_name(self, name: str) -> bool:
        """Add a name to the blacklist"""
        try:
            is_valid, message = SecurityValidator.validate_input(name)
            if not is_valid:
                logger.warning(f"Invalid name rejected: {message}")
                return False
            
            cleaned_name = name.strip()
            normalized = arabic_processor.normalize_arabic_text(cleaned_name)
            
            with self.lock:
                if cleaned_name not in self.blacklist:
                    self.blacklist.append(cleaned_name)
                    self.normalized_blacklist[cleaned_name] = normalized
                    self.version += 1
                    
                    # Update CSV file
                    self._save_to_csv()
                    logger.info(f"Added name to blacklist: {cleaned_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding name to blacklist: {e}")
            return False
    
    def remove_name(self, name: str) -> bool:
        """Remove a name from the blacklist"""
        try:
            with self.lock:
                if name in self.blacklist:
                    self.blacklist.remove(name)
                    self.normalized_blacklist.pop(name, None)
                    self.version += 1
                    
                    # Update CSV file
                    self._save_to_csv()
                    logger.info(f"Removed name from blacklist: {name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing name from blacklist: {e}")
            return False
    
    def _save_to_csv(self):
        """Save current blacklist to CSV file"""
        try:
            df = pd.DataFrame({'name': self.blacklist})
            df.to_csv(self.csv_path, index=False)
            self.last_modified = time.time()
        except Exception as e:
            logger.error(f"Error saving blacklist to CSV: {e}")
    
    def get_blacklist(self) -> List[str]:
        """Get current blacklist thread-safely"""
        with self.lock:
            return self.blacklist.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """Get blacklist information"""
        with self.lock:
            return {
                "count": len(self.blacklist),
                "version": self.version,
                "last_modified": self.last_modified,
                "file_path": self.csv_path
            }

# Initialize enhanced components
arabic_processor = ArabicTextProcessor()
similarity_calculator = AdvancedSimilarityCalculator()
performance_monitor = AdvancedPerformanceMonitor()
rate_limiter = RateLimiter()
security_validator = SecurityValidator()

# Initialize dynamic blacklist manager
csv_path = os.path.join(os.path.dirname(__file__), "blacklist.csv")
blacklist_manager = DynamicBlacklistManager(csv_path)

# Start monitoring
performance_monitor.start_monitoring()

# Enhanced Request/Response Models
class NameCheckRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=Config.MAX_INPUT_LENGTH, 
                     description="Name to check against blacklist")
    
    @validator('name')
    def validate_name(cls, v):
        is_valid, message = SecurityValidator.validate_input(v)
        if not is_valid:
            raise ValueError(message)
        return SecurityValidator.sanitize_input(v)

class BatchNameCheckRequest(BaseModel):
    names: List[str] = Field(..., min_items=1, max_items=Config.MAX_BATCH_SIZE,
                            description="List of names to check")
    
    @validator('names')
    def validate_names(cls, v):
        validated_names = []
        for name in v:
            is_valid, message = SecurityValidator.validate_input(name)
            if not is_valid:
                raise ValueError(f"Invalid name '{name}': {message}")
            validated_names.append(SecurityValidator.sanitize_input(name))
        return validated_names

class BlacklistUpdateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100,
                     description="Name to add or remove from blacklist")
    action: str = Field(..., description="Action to perform: 'add' or 'remove'")
    
    @validator('action')
    def validate_action(cls, v):
        if v not in ['add', 'remove']:
            raise ValueError("Action must be 'add' or 'remove'")
        return v

class EnhancedSimilarityResult(BaseModel):
    blacklisted_name: str
    similarity_percentage: float
    similarity_details: Optional[Dict[str, float]] = None
    normalized_input: Optional[str] = None
    normalized_blacklist: Optional[str] = None
    confidence_level: Optional[str] = None
    language_detected: Optional[str] = None

class EnhancedNameCheckResponse(BaseModel):
    input_name: str
    normalized_input: str
    is_suspicious: bool
    max_similarity: float
    confidence_level: str
    language_detected: str
    matches: List[EnhancedSimilarityResult]
    processing_info: Dict[str, Any] = {}
    security_info: Dict[str, Any] = {}
    
class BatchNameCheckResponse(BaseModel):
    results: List[EnhancedNameCheckResponse]
    summary: Dict[str, Any]
    processing_time: float

# Middleware for rate limiting and logging
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting and request logging middleware"""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        performance_monitor.blocked_requests += 1
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later."}
        )
    
    # Log request
    start_time = time.time()
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # Log successful request
        logger.info(f"Request from {client_ip} processed in {processing_time:.4f}s")
        
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.error_count += 1
        logger.error(f"Error processing request from {client_ip}: {e}")
        raise

# Add middleware
app.middleware("http")(rate_limit_middleware)

@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive API information"""
    # Check blacklist status
    blacklist_status = blacklist_manager.get_info()
    
    return {
        "message": "Advanced Arabic Names Detection API",
        "description": "Enterprise-grade Arabic name similarity detection with security, performance, and analytics",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "✅ Advanced Arabic text normalization and diacritics handling",
            "✅ Multi-algorithm phonetic similarity matching", 
            "✅ Context-aware token-based analysis",
            "✅ Weighted similarity scoring with false positive reduction",
            "✅ Transliteration support (Arabic ⟷ Latin)",
            "✅ Real-time confidence level assessment",
            "✅ Dynamic blacklist management",
            "✅ Comprehensive security validation",
            "✅ Rate limiting and performance monitoring",
            "✅ Batch processing capabilities",
            "✅ Historical analytics and pattern detection"
        ],
        "endpoints": {
            "POST /check-name": "Enhanced comprehensive name analysis",
            "POST /check-names-batch": "Batch name checking for multiple inputs",
            "POST /check-name-simple": "Quick name check with advanced processing",
            "GET /blacklist": "Get current blacklist with metadata",
            "POST /blacklist/update": "Add or remove names from blacklist",
            "GET /config": "Get current API configuration",
            "POST /config": "Update API configuration",
            "GET /health": "Detailed health check and system status",
            "GET /stats": "Advanced performance statistics and analytics",
            "GET /analytics": "Usage analytics and pattern insights"
        },
        "blacklist_info": blacklist_status,
        "security_features": [
            "Input validation and sanitization",
            "Rate limiting protection", 
            "Injection attack prevention",
            "Comprehensive logging"
        ]
    }

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with comprehensive system status"""
    # Check blacklist auto-reload
    blacklist_manager.reload_if_changed()
    
    blacklist_info = blacklist_manager.get_info()
    cache_info = {
        "normalization": arabic_processor.normalize_arabic_text.cache_info()._asdict(),
        "tokens": arabic_processor.get_arabic_tokens.cache_info()._asdict(),
        "phonetic": arabic_processor.get_phonetic_code.cache_info()._asdict(),
        "similarity": len(similarity_calculator._similarity_cache)
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "blacklist": {
            "loaded": blacklist_info["count"] > 0,
            "count": blacklist_info["count"], 
            "version": blacklist_info["version"],
            "last_modified": blacklist_info["last_modified"]
        },
        "cache_status": cache_info,
        "performance": performance_monitor.get_detailed_stats(),
        "system_resources": {
            "memory_usage": "Available", # Could add psutil for detailed monitoring
            "cpu_usage": "Available"
        }
    }
    """Health check endpoint"""
    return {
        "status": "healthy",
        "blacklist": {
            "loaded": len(blacklist_manager.get_blacklist()) > 0,
            "count": len(blacklist_manager.get_blacklist()),
            "version": blacklist_manager.version,
            "last_modified": blacklist_manager.last_modified
        },
        "cache_info": {
            "normalization_cache_size": arabic_processor.normalize_arabic_text.cache_info()._asdict(),
            "tokens_cache_size": arabic_processor.get_arabic_tokens.cache_info()._asdict(),
            "phonetic_cache_size": arabic_processor.get_phonetic_code.cache_info()._asdict()
        }
    }

@app.get("/stats")
async def get_performance_stats():
    """Get performance statistics"""
    return {
        "performance": performance_monitor.get_stats(),
        "cache_stats": {
            "text_normalization": arabic_processor.normalize_arabic_text.cache_info()._asdict(),
            "token_extraction": arabic_processor.get_arabic_tokens.cache_info()._asdict(),
            "phonetic_coding": arabic_processor.get_phonetic_code.cache_info()._asdict(),
            "similarity_cache_size": len(similarity_calculator._similarity_cache)
        },
        "configuration": {
            "min_threshold": Config.MIN_SIMILARITY_THRESHOLD,
            "suspicious_threshold": Config.SUSPICIOUS_THRESHOLD,
            "max_results": Config.MAX_RESULTS
        }
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

@app.post("/check-name", response_model=EnhancedNameCheckResponse)
async def enhanced_check_name(request: NameCheckRequest):
    """
    Enhanced Arabic name similarity detection with comprehensive analysis
    Features: Advanced text processing, security validation, and detailed analytics
    """
    start_time = time.time()
    
    try:
        # Get current blacklist (auto-reloads if changed)
        blacklist_manager.reload_if_changed()
        current_blacklist = blacklist_manager.get_blacklist()
        
        if not current_blacklist:
            raise HTTPException(status_code=500, detail="Blacklist not loaded or empty")
        
        input_name = request.name
        
        # Detect language
        language = arabic_processor.detect_language(input_name)
        
        # Normalize input name
        normalized_input = arabic_processor.normalize_arabic_text(input_name)
        
        if not normalized_input:
            raise HTTPException(status_code=400, detail="Invalid input: normalization resulted in empty string")
        
        # Initialize tracking
        similarities = []
        processing_stats = {
            "total_comparisons": str(len(current_blacklist)),
            "above_threshold": "0",
            "cache_hits": "0",
            "processing_time": "0",
            "language_detected": language
        }
        
        # Process each blacklisted name
        for blacklisted_name in current_blacklist:
            # Check cache first
            cache_key = similarity_calculator._get_cache_key(normalized_input, blacklisted_name)
            cache_hit = cache_key in similarity_calculator._similarity_cache
            
            if cache_hit:
                processing_stats["cache_hits"] = str(int(processing_stats["cache_hits"]) + 1)
            
            # Calculate comprehensive similarity
            similarity_scores = similarity_calculator.calculate_comprehensive_similarity(
                normalized_input, blacklisted_name
            )
            
            # Get weighted final score
            final_score = similarity_calculator.get_weighted_similarity(similarity_scores)
            
            if final_score >= Config.MIN_SIMILARITY_THRESHOLD:
                processing_stats["above_threshold"] = str(int(processing_stats["above_threshold"]) + 1)
                
                # Determine confidence level for this match
                if final_score >= 95:
                    match_confidence = "very_high"
                elif final_score >= 85:
                    match_confidence = "high"
                elif final_score >= 70:
                    match_confidence = "medium"
                elif final_score >= 60:
                    match_confidence = "low"
                else:
                    match_confidence = "very_low"
                
                similarities.append(EnhancedSimilarityResult(
                    blacklisted_name=blacklisted_name,
                    similarity_percentage=final_score,
                    similarity_details=similarity_scores,
                    normalized_input=normalized_input,
                    normalized_blacklist=arabic_processor.normalize_arabic_text(blacklisted_name),
                    confidence_level=match_confidence,
                    language_detected=language
                ))
        
        # Sort by similarity percentage (descending)
        similarities.sort(key=lambda x: x.similarity_percentage, reverse=True)
        
        # Determine overall suspicion level and confidence
        max_sim = similarities[0].similarity_percentage if similarities else 0
        is_suspicious = max_sim >= Config.SUSPICIOUS_THRESHOLD
        
        # Enhanced confidence calculation
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
        
        # Calculate processing time
        processing_time = time.time() - start_time
        processing_stats["processing_time"] = f"{processing_time:.4f}"
        
        # Security information
        security_info = {
            "input_sanitized": input_name != request.name,
            "language_detected": language,
            "normalized_length": len(normalized_input),
            "original_length": len(input_name)
        }
        
        # Record metrics
        performance_monitor.record_request(
            request_data={"name": input_name[:50]}, # Truncated for privacy
            comparisons=len(current_blacklist),
            cache_hit=int(processing_stats["cache_hits"]) > 0
        )
        
        if is_suspicious:
            performance_monitor.record_suspicious_pattern(normalized_input[:20])  # Truncated pattern
        
        return EnhancedNameCheckResponse(
            input_name=input_name,
            normalized_input=normalized_input,
            is_suspicious=is_suspicious,
            max_similarity=max_sim,
            confidence_level=confidence,
            language_detected=language,
            matches=similarities[:Config.MAX_RESULTS],
            processing_info=processing_stats,
            security_info=security_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        performance_monitor.error_count += 1
        logger.error(f"Error in enhanced_check_name: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/check-names-batch", response_model=BatchNameCheckResponse)
async def check_names_batch(request: BatchNameCheckRequest):
    """
    Batch processing endpoint for checking multiple names efficiently
    """
    start_time = time.time()
    
    try:
        results = []
        suspicious_count = 0
        total_matches = 0
        
        # Process each name
        for name in request.names:
            name_request = NameCheckRequest(name=name)
            result = await enhanced_check_name(name_request)
            results.append(result)
            
            if result.is_suspicious:
                suspicious_count += 1
                total_matches += len(result.matches)
        
        processing_time = time.time() - start_time
        
        summary = {
            "total_names": len(request.names),
            "suspicious_names": suspicious_count,
            "safe_names": len(request.names) - suspicious_count,
            "total_matches": total_matches,
            "average_processing_time": processing_time / len(request.names)
        }
        
        return BatchNameCheckResponse(
            results=results,
            summary=summary,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")
    if not normalized_input:
        raise HTTPException(status_code=400, detail="Invalid Arabic name provided")
    
    # Calculate similarities with each blacklisted name
    similarities = []
    processing_stats = {
        "total_comparisons": str(len(BLACKLIST)),
        "above_threshold": "0",
        "processing_method": "advanced_arabic_nlp"
    }
    
    # Record performance metrics
    performance_monitor.record_request(len(BLACKLIST))
    
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
    # Get current blacklist (auto-reloads if changed)
    blacklist_manager.reload_if_changed()
    current_blacklist = blacklist_manager.get_blacklist()
    
    if not current_blacklist:
        raise HTTPException(status_code=500, detail="Blacklist not loaded")
    
    input_name = request.name
    
    # Normalize input
    normalized_input = arabic_processor.normalize_arabic_text(input_name)
    
    # Find best match using advanced similarity
    best_score = 0
    best_match = None
    best_details = {}
    
    for blacklisted_name in current_blacklist:
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

@app.get("/blacklist")
async def get_enhanced_blacklist():
    """Get current blacklist with comprehensive metadata"""
    blacklist_info = blacklist_manager.get_info()
    current_blacklist = blacklist_manager.get_blacklist()
    
    # Generate statistics
    stats = {
        "total_names": len(current_blacklist),
        "average_length": sum(len(name) for name in current_blacklist) / len(current_blacklist) if current_blacklist else 0,
        "unique_first_letters": len(set(name[0] for name in current_blacklist if name)),
        "contains_arabic": sum(1 for name in current_blacklist if arabic_processor.detect_language(name) == "arabic"),
        "contains_mixed": sum(1 for name in current_blacklist if arabic_processor.detect_language(name) == "mixed")
    }
    
    return {
        "blacklist": current_blacklist,
        "metadata": blacklist_info,
        "statistics": stats,
        "last_updated": datetime.fromtimestamp(blacklist_info["last_modified"]).isoformat() if blacklist_info["last_modified"] else None
    }

@app.post("/blacklist/update")
async def update_blacklist(request: BlacklistUpdateRequest):
    """Add or remove names from blacklist with validation"""
    try:
        if request.action == "add":
            success = blacklist_manager.add_name(request.name)
            if success:
                return {
                    "status": "success",
                    "action": "added",
                    "name": request.name,
                    "blacklist_version": blacklist_manager.version,
                    "total_names": len(blacklist_manager.get_blacklist())
                }
            else:
                return {
                    "status": "failed",
                    "action": "add",
                    "name": request.name,
                    "reason": "Name already exists or validation failed"
                }
        
        elif request.action == "remove":
            success = blacklist_manager.remove_name(request.name)
            if success:
                return {
                    "status": "success",
                    "action": "removed",
                    "name": request.name,
                    "blacklist_version": blacklist_manager.version,
                    "total_names": len(blacklist_manager.get_blacklist())
                }
            else:
                return {
                    "status": "failed",
                    "action": "remove",
                    "name": request.name,
                    "reason": "Name not found in blacklist"
                }
                
    except Exception as e:
        logger.error(f"Error updating blacklist: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating blacklist: {str(e)}")

@app.get("/stats")
async def get_enhanced_stats():
    """Get comprehensive performance statistics and analytics"""
    stats = performance_monitor.get_detailed_stats()
    
    cache_info = {
        "text_normalization": arabic_processor.normalize_arabic_text.cache_info()._asdict(),
        "token_extraction": arabic_processor.get_arabic_tokens.cache_info()._asdict(),
        "phonetic_coding": arabic_processor.get_phonetic_code.cache_info()._asdict(),
        "similarity_cache_size": len(similarity_calculator._similarity_cache),
        "similarity_cache_max": Config.CACHE_SIZE
    }
    
    blacklist_info = blacklist_manager.get_info()
    
    return {
        "performance": stats,
        "cache_statistics": cache_info,
        "blacklist_info": blacklist_info,
        "configuration": {
            "min_similarity_threshold": Config.MIN_SIMILARITY_THRESHOLD,
            "suspicious_threshold": Config.SUSPICIOUS_THRESHOLD,
            "max_results": Config.MAX_RESULTS,
            "max_input_length": Config.MAX_INPUT_LENGTH,
            "rate_limit": Config.REQUEST_RATE_LIMIT
        },
        "system_info": {
            "version": "3.0.0",
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/analytics")
async def get_analytics():
    """Get usage analytics and pattern insights"""
    stats = performance_monitor.get_detailed_stats()
    
    # Generate insights
    insights = []
    if stats["cache_hit_rate"] > 80:
        insights.append("High cache efficiency - system is performing well")
    elif stats["cache_hit_rate"] < 30:
        insights.append("Low cache efficiency - consider increasing cache size")
    
    if stats["error_count"] > stats["total_requests"] * 0.05:
        insights.append("High error rate detected - investigate system stability")
    
    if stats["blocked_requests"] > 0:
        insights.append(f"{stats['blocked_requests']} requests blocked due to rate limiting")
    
    return {
        "summary": {
            "total_requests": stats["total_requests"],
            "success_rate": (stats["total_requests"] - stats["error_count"]) / max(stats["total_requests"], 1) * 100,
            "cache_efficiency": stats["cache_hit_rate"],
            "average_response_time": stats.get("avg_processing_time", 0)
        },
        "patterns": {
            "suspicious_patterns": stats["suspicious_patterns"],
            "most_common_pattern": max(stats["suspicious_patterns"].items(), key=lambda x: x[1]) if stats["suspicious_patterns"] else None
        },
        "insights": insights,
        "recommendations": [
            "Monitor cache hit ratio for optimal performance",
            "Review suspicious patterns for potential false positives",
            "Consider blacklist updates based on usage patterns"
        ]
    }

@app.post("/config")
async def update_configuration(config_update: dict):
    """Update API configuration with validation"""
    try:
        updated_fields = []
        
        if "min_similarity_threshold" in config_update:
            value = float(config_update["min_similarity_threshold"])
            if 0 <= value <= 100:
                Config.MIN_SIMILARITY_THRESHOLD = value
                updated_fields.append("min_similarity_threshold")
        
        if "suspicious_threshold" in config_update:
            value = float(config_update["suspicious_threshold"])
            if 0 <= value <= 100:
                Config.SUSPICIOUS_THRESHOLD = value
                updated_fields.append("suspicious_threshold")
        
        if "max_results" in config_update:
            value = int(config_update["max_results"])
            if 1 <= value <= 100:
                Config.MAX_RESULTS = value
                updated_fields.append("max_results")
        
        if "request_rate_limit" in config_update:
            value = int(config_update["request_rate_limit"])
            if 1 <= value <= 1000:
                Config.REQUEST_RATE_LIMIT = value
                updated_fields.append("request_rate_limit")
        
        logger.info(f"Configuration updated: {updated_fields}")
        
        return {
            "status": "success",
            "updated_fields": updated_fields,
            "current_config": {
                "min_similarity_threshold": Config.MIN_SIMILARITY_THRESHOLD,
                "suspicious_threshold": Config.SUSPICIOUS_THRESHOLD,
                "max_results": Config.MAX_RESULTS,
                "request_rate_limit": Config.REQUEST_RATE_LIMIT
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)