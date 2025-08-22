# Names Detection API

An Arabic name similarity detection system built with FastAPI that checks names against a blacklist and returns similarity percentages.

## Description

This API allows you to check if a given Arabic name is similar to any name in a predefined blacklist. It uses fuzzy string matching algorithms to calculate similarity percentages and determine if a name should be considered suspicious.

## Features

- **FastAPI-based REST API** with automatic documentation
- **Arabic text support** for name matching
- **Multiple similarity algorithms** (ratio, partial ratio, token sort, token set)
- **Configurable similarity thresholds**
- **CSV-based blacklist** for easy management
- **Multiple endpoints** for different use cases

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
