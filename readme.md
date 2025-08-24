# Professional Arabic Names Detection API

An advanced Arabic name similarity detection system built with FastAPI that uses sophisticated text processing and multiple similarity algorithms to check names against a comprehensive blacklist.

## Description

This professional-grade API provides comprehensive Arabic name similarity detection with advanced features designed specifically for Arabic text processing. It uses state-of-the-art natural language processing techniques to handle the complexities of Arabic script, including diacritics, text normalization, phonetic matching, and various spelling variations.

## Key Features

- **Advanced Arabic Text Processing** with proper normalization and diacritics handling
- **Multiple Similarity Algorithms** including phonetic, token-based, and fuzzy matching
- **Weighted Scoring System** for more accurate similarity assessment
- **Configurable Thresholds** for fine-tuning detection sensitivity
- **Comprehensive Blacklist** with 39+ carefully curated names
- **Confidence Level Assessment** for reliability evaluation
- **Real-time Configuration** management via API endpoints
- **Detailed Similarity Analysis** with breakdown of different matching methods
- **Performance Optimized** for production environments
- **Professional API Design** with comprehensive documentation

## Technical Improvements

### Arabic Text Normalization
- Automatic diacritics (tashkeel) removal
- Alef form normalization (أ، إ، آ → ا)
- Teh marbuta normalization (ة → ه)
- Tatweel (kashida) removal
- Extra whitespace cleanup
- Non-Arabic character filtering

### Advanced Similarity Algorithms
- **Exact Matching**: Perfect character-by-character comparison
- **Normalized Matching**: Comparison after Arabic text normalization
- **Token-based Analysis**: Individual word/token comparison
- **Phonetic Matching**: Sound-based similarity for Arabic names
- **Fuzzy String Matching**: Multiple fuzzy algorithms (ratio, partial, token sort/set)
- **Weighted Final Score**: Intelligent combination of all similarity metrics

### Configuration Management
- Adjustable similarity thresholds
- Configurable suspicious detection levels
- Customizable result limits
- Runtime configuration updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akhaly998/namesDetection.git
cd namesDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: Lightning-fast ASGI server
- **PyArabic**: Comprehensive Arabic text processing
- **Arabic Reshaper**: Proper Arabic text rendering
- **Python-BIDI**: Bidirectional text support
- **Jellyfish**: Advanced phonetic matching algorithms
- **RapidFuzz**: High-performance fuzzy string matching
- **FuzzyWuzzy**: Traditional fuzzy string matching
- **Python-Levenshtein**: Fast string distance calculations
- **Pandas**: Efficient data handling for blacklist management

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Check Name (Detailed)
**POST** `/check-name`

Returns detailed similarity analysis with all matches above threshold.

```bash
curl -X POST "http://localhost:8000/check-name" \
     -H "Content-Type: application/json" \
     -d '{"name": "صدام حسن سعيد مجيد"}'
```

Response:
```json
{
  "input_name": "صدام حسن سعيد مجيد",
  "is_suspicious": true,
  "max_similarity": 100.0,
  "matches": [
    {
      "blacklisted_name": "صدام حسن سعيد",
      "similarity_percentage": 100.0
    }
  ]
}
```

#### 2. Check Name (Simple)
**POST** `/check-name-simple`

Returns the best match only.

```bash
curl -X POST "http://localhost:8000/check-name-simple" \
     -H "Content-Type: application/json" \
     -d '{"name": "صدام حسن سعيد مجيد"}'
```

#### 3. Get Blacklist
**GET** `/blacklist`

Returns the current blacklist.

#### 4. Health Check
**GET** `/health`

Returns API health status.

## Configuration

### Similarity Thresholds

The API uses the following thresholds:
- **Minimum threshold**: 60% (names below this are not returned)
- **Suspicious threshold**: 70% (names above this are marked as suspicious)

### Blacklist Management

The blacklist is stored in `blacklist.csv`. To add or remove names:

1. Edit the `blacklist.csv` file
2. Restart the server to reload the blacklist

## Examples

### Testing with Example Names

```bash
# Test with the example from requirements
curl -X POST "http://localhost:8000/check-name-simple" \
     -H "Content-Type: application/json" \
     -d '{"name": "صدام حسن سعيد مجيد"}'

# Test with a safe name
curl -X POST "http://localhost:8000/check-name-simple" \
     -H "Content-Type: application/json" \
     -d '{"name": "محمد علي"}'

# Test with a close match
curl -X POST "http://localhost:8000/check-name-simple" \
     -H "Content-Type: application/json" \
     -d '{"name": "اسامة بن لادن"}'
```

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **FuzzyWuzzy**: String matching library
- **python-Levenshtein**: Fast string similarity calculations
- **Pandas**: CSV data handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
