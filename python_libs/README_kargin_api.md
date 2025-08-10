# Kargin Haghordum API 🎭

A comprehensive FastAPI application for searching and managing Kargin Haghordum comedy sketches based on the CSV data.

## Features

- ✅ **Health Check**: Monitor API status and data statistics
- 🔍 **Text Search**: Search through sketch titles, transcripts, and metadata
- 🎲 **Random Sketches**: Get random sketch recommendations
- ✏️ **Data Updates**: Update sketch information
- 📊 **Statistics**: Get insights about the sketch collection

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_kargin_api.txt
```

2. Make sure the `kargin.csv` file is in the `assets/` directory relative to the API file.

## Running the API

Start the development server:
```bash
python kargin_api.py
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check 🩺
```http
GET /health
```

Returns the current status of the API and basic statistics.

**Example Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-10T12:00:00",
  "message": "Kargin API is running smoothly! 🎭✨",
  "total_sketches": 95
}
```

### 2. Text Search 🔍
```http
GET /search?q={query}&limit={limit}&include_incomplete={bool}
```

**Parameters:**
- `q`: Search query (required)
- `limit`: Maximum results (1-100, default: 10)
- `include_incomplete`: Include unfinished sketches (default: true)

**Example:**
```http
GET /search?q=բժիշկ&limit=5
```

**Example Response:**
```json
{
  "query": "բժիշկ",
  "matches_found": 12,
  "results": [
    {
      "index": 25,
      "title": "Kargin Haghordum sketch 442 (Hayko Mko)",
      "link": "https://www.youtube.com/watch?v=xVpyEOc0y_Y",
      "text_common": "Սուրեն ես քո վրա զարմանում եմ...",
      "main_actors": "Հակյո, Մկո",
      "main_actors_count": 2,
      "roles_names": "Սուրեն, Արգամվիչ",
      "location": "Հիվանդանոց",
      "lighting": "լուսավոր",
      "languages": "հայերեն+ռուսերեն",
      "done": 1.0
    }
  ]
}
```

### 3. Random Sketch 🎲
```http
GET /random?completed_only={bool}
```

**Parameters:**
- `completed_only`: Only return completed sketches (default: true)

**Example Response:**
```json
{
  "sketch": {
    "index": 45,
    "title": "Kargin Haghordum sketch 489 (Hayko Mko)",
    "link": "https://www.youtube.com/watch?v=M1hWImC8Vl4",
    "main_actors": "Հայկո, Մկո, Աշոտ",
    "location": "Դուրս",
    "done": 1.0
  },
  "message": "🎲 Random sketch: Kargin Haghordum sketch 489 (Hayko Mko)"
}
```

### 4. Update Sketch Data ✏️
```http
PUT /sketches/{sketch_index}
```

**Request Body:**
```json
{
  "title": "New Title",
  "text_common": "New common text",
  "main_actors": "New actors list",
  "done": 1.0
}
```

**Example Response:**
```json
{
  "success": true,
  "message": "Successfully updated fields: title, text_common",
  "updated_sketch": {
    "index": 0,
    "title": "New Title",
    "main_actors": "New actors list"
  }
}
```

### 5. Get Specific Sketch 📄
```http
GET /sketches/{sketch_index}
```

Returns detailed information about a specific sketch by its index.

### 6. Statistics 📊
```http
GET /stats
```

**Example Response:**
```json
{
  "total_sketches": 95,
  "completed_sketches": 85,
  "incomplete_sketches": 10,
  "unique_locations": 12,
  "most_common_location": "Տուն",
  "completion_rate": 89.5,
  "average_actors_per_sketch": 2.3
}
```

## Testing the API

Run the example test script:
```bash
python test_kargin_api.py
```

This will test all endpoints and show example usage.

## Usage Examples

### Python with requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Search for sketches
response = requests.get("http://localhost:8000/search", 
                       params={"q": "բժիշկ", "limit": 3})
results = response.json()
for sketch in results['results']:
    print(f"Title: {sketch['title']}")
    print(f"Link: {sketch['link']}")

# Get random sketch
response = requests.get("http://localhost:8000/random")
random_sketch = response.json()
print(f"Random sketch: {random_sketch['sketch']['title']}")
```

### JavaScript with fetch
```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Search
fetch('http://localhost:8000/search?q=բժիշկ&limit=5')
  .then(response => response.json())
  .then(data => {
    console.log(`Found ${data.matches_found} matches`);
    data.results.forEach(sketch => {
      console.log(`- ${sketch.title}`);
    });
  });
```

### curl
```bash
# Health check
curl "http://localhost:8000/health"

# Search
curl "http://localhost:8000/search?q=բժիշկ&limit=3"

# Random sketch
curl "http://localhost:8000/random"

# Update sketch
curl -X PUT "http://localhost:8000/sketches/0" \
     -H "Content-Type: application/json" \
     -d '{"text_common": "Updated text"}'
```

## Data Structure

The API works with CSV data containing:
- `titles`: Sketch titles
- `links`: YouTube links
- `text_common`: Short text excerpts
- `text`: Full transcripts
- `main_actors`: Actor names
- `main_actors_count`: Number of actors
- `roles_names`: Character names
- `location`: Sketch location/setting
- `lighting`: Lighting conditions
- `languages`: Languages used
- `done`: Completion status (0.0 or 1.0)

## Error Handling

The API provides detailed error messages:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (sketch doesn't exist)
- `422`: Validation Error (invalid data format)
- `500`: Internal Server Error

## Development

The API uses:
- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and serialization
- **pandas**: Data manipulation
- **uvicorn**: ASGI server

## Contributing

1. Fork the repository
2. Make your changes
3. Test with `python test_kargin_api.py`
4. Submit a pull request

## License

This project is for educational purposes as part of a Python course.

---

🎭 **Enjoy exploring the Kargin Haghordum sketches!** 🎭
