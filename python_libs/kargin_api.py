"""
Kargin Haghordum Sketches API

A comprehensive FastAPI application that provides endpoints for:
1. Health check
2. Text search in sketch transcripts
3. Getting random sketch links
4. Updating sketch data

Data source: kargin.csv containing Kargin Haghordum sketch information
"""

import pandas as pd
import random
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator


# Pydantic Models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime
    message: str = "Kargin API is running smoothly"
    total_sketches: int


class SketchInfo(BaseModel):
    """Basic sketch information model"""
    index: int
    title: str
    link: str
    text_common: Optional[str] = None
    text: Optional[str] = None
    main_actors: str
    main_actors_count: int
    roles_names: str
    location: str
    lighting: str
    languages: str
    done: float


class SearchResponse(BaseModel):
    """Search results response model"""
    query: str
    matches_found: int
    results: List[SketchInfo]
    search_fields: List[str] = ["title", "text_common", "text", "roles_names"]


class RandomSketchResponse(BaseModel):
    """Random sketch response model"""
    sketch: SketchInfo
    message: str = "Here's a random Kargin sketch for you!"


class UpdateSketchRequest(BaseModel):
    """Request model for updating sketch data"""
    index: int = Field(..., description="Index of the sketch to update")
    title: Optional[str] = Field(None, description="New title")
    text_common: Optional[str] = Field(None, description="New common text")
    text: Optional[str] = Field(None, description="New full text")
    main_actors: Optional[str] = Field(None, description="New main actors")
    main_actors_count: Optional[int] = Field(None, description="New actor count")
    roles_names: Optional[str] = Field(None, description="New role names")
    location: Optional[str] = Field(None, description="New location")
    lighting: Optional[str] = Field(None, description="New lighting")
    languages: Optional[str] = Field(None, description="New languages")
    done: Optional[float] = Field(None, description="Completion status (0.0 or 1.0)")
    
    @validator('done')
    def validate_done(cls, v):
        if v is not None and v not in [0.0, 1.0]:
            raise ValueError('done must be either 0.0 or 1.0')
        return v


class UpdateResponse(BaseModel):
    """Update operation response model"""
    success: bool
    message: str
    updated_sketch: Optional[SketchInfo] = None


# Global variable to store data
sketches_df: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown"""
    # Startup
    load_data()
    print("🚀 Kargin API started successfully!")
    yield
    # Shutdown (if needed)
    print("🛑 Kargin API shutting down...")


def load_data() -> pd.DataFrame:
    """Load the Kargin sketches data from CSV"""
    global sketches_df
    if sketches_df is None:
        csv_path = Path(__file__).parent / "assets" / "kargin.csv"
        try:
            sketches_df = pd.read_csv(csv_path)
            # Fill NaN values with empty strings for text fields
            text_columns = ['text_common', 'text', 'roles_names']
            for col in text_columns:
                if col in sketches_df.columns:
                    sketches_df[col] = sketches_df[col].fillna('')
            
            print(f"✅ Loaded {len(sketches_df)} sketches from {csv_path}")
        except FileNotFoundError:
            raise HTTPException(
                status_code=500, 
                detail=f"Data file not found at {csv_path}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading data: {str(e)}"
            )
    return sketches_df


def get_data_dependency():
    """FastAPI dependency to ensure data is loaded"""
    return load_data()


# Initialize FastAPI app
app = FastAPI(
    title="Kargin Haghordum API",
    description="API for searching and managing Kargin Haghordum comedy sketches",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


def dataframe_row_to_sketch_info(row: pd.Series, index: int) -> SketchInfo:
    """Convert a DataFrame row to SketchInfo model"""
    return SketchInfo(
        index=index,
        title=str(row['titles']),
        link=str(row['links']),
        text_common=str(row['text_common']) if pd.notna(row['text_common']) else None,
        text=str(row['text']) if pd.notna(row['text']) else None,
        main_actors=str(row['main_actors']),
        main_actors_count=int(row['main_actors_count']),
        roles_names=str(row['roles_names']),
        location=str(row['location']),
        lighting=str(row['lighting']),
        languages=str(row['languages']),
        done=float(row['done'])
    )


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Kargin Haghordum API! 🎭",
        "description": "Search and explore Armenian comedy sketches",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(df: pd.DataFrame = Depends(get_data_dependency)):
    """
    Health check endpoint
    
    Returns the current status of the API and basic statistics about the data.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        message="Kargin API is running smoothly! 🎭✨",
        total_sketches=len(df)
    )


@app.get("/search", response_model=SearchResponse)
async def search_sketches(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=100),
    include_incomplete: bool = Query(True, description="Include incomplete sketches (done=0)"),
    df: pd.DataFrame = Depends(get_data_dependency)
):
    """
    Search sketches based on text content
    
    Searches through:
    - Sketch titles
    - Common text (text_common)
    - Full text transcripts
    - Role names
    
    Returns matching sketches with relevance scoring.
    """
    # Filter by completion status if needed
    search_df = df if include_incomplete else df[df['done'] == 1.0]
    
    if search_df.empty:
        return SearchResponse(
            query=q,
            matches_found=0,
            results=[]
        )
    
    # Define searchable columns
    search_columns = ['titles', 'text_common', 'text', 'roles_names']
    
    # Create a comprehensive search
    query_lower = q.lower()
    matches = []
    
    for idx, row in search_df.iterrows():
        relevance_score = 0
        match_found = False
        
        # Search in each column with different weights
        column_weights = {
            'titles': 3.0,      # Title matches are most important
            'text_common': 2.0,  # Common text is quite important
            'text': 1.0,        # Full text matches
            'roles_names': 1.5   # Role names are moderately important
        }
        
        for col in search_columns:
            if col in row and pd.notna(row[col]):
                text_content = str(row[col]).lower()
                
                # Count occurrences of search term
                occurrences = len(re.findall(re.escape(query_lower), text_content))
                if occurrences > 0:
                    match_found = True
                    relevance_score += occurrences * column_weights[col]
        
        if match_found:
            sketch_info = dataframe_row_to_sketch_info(row, int(idx))
            matches.append((relevance_score, sketch_info))
    
    # Sort by relevance score (descending) and limit results
    matches.sort(key=lambda x: x[0], reverse=True)
    results = [sketch for _, sketch in matches[:limit]]
    
    return SearchResponse(
        query=q,
        matches_found=len(matches),
        results=results
    )


@app.get("/random", response_model=RandomSketchResponse)
async def get_random_sketch(
    completed_only: bool = Query(True, description="Only return completed sketches"),
    df: pd.DataFrame = Depends(get_data_dependency)
):
    """
    Get a random sketch link
    
    Returns a randomly selected sketch. By default, only returns completed sketches.
    """
    # Filter data based on completion status
    available_sketches = df[df['done'] == 1.0] if completed_only else df
    
    if available_sketches.empty:
        raise HTTPException(
            status_code=404,
            detail="No sketches available with the specified criteria"
        )
    
    # Select random sketch
    random_row = available_sketches.sample(n=1).iloc[0]
    random_index = available_sketches.sample(n=1).index[0]
    
    sketch = dataframe_row_to_sketch_info(random_row, int(random_index))
    
    return RandomSketchResponse(
        sketch=sketch,
        message=f"🎲 Random sketch: {sketch.title}"
    )


@app.put("/sketches/{sketch_index}", response_model=UpdateResponse)
async def update_sketch(
    sketch_index: int,
    update_data: UpdateSketchRequest,
    df: pd.DataFrame = Depends(get_data_dependency)
):
    """
    Update sketch data
    
    Updates specific fields of a sketch by its index.
    Only provided fields will be updated.
    """
    global sketches_df
    
    # Validate sketch index
    if sketch_index not in df.index:
        raise HTTPException(
            status_code=404,
            detail=f"Sketch with index {sketch_index} not found"
        )
    
    # Update fields that are provided
    updates_made = []
    
    if update_data.title is not None:
        sketches_df.at[sketch_index, 'titles'] = update_data.title
        updates_made.append("title")
    
    if update_data.text_common is not None:
        sketches_df.at[sketch_index, 'text_common'] = update_data.text_common
        updates_made.append("text_common")
    
    if update_data.text is not None:
        sketches_df.at[sketch_index, 'text'] = update_data.text
        updates_made.append("text")
    
    if update_data.main_actors is not None:
        sketches_df.at[sketch_index, 'main_actors'] = update_data.main_actors
        updates_made.append("main_actors")
    
    if update_data.main_actors_count is not None:
        sketches_df.at[sketch_index, 'main_actors_count'] = update_data.main_actors_count
        updates_made.append("main_actors_count")
    
    if update_data.roles_names is not None:
        sketches_df.at[sketch_index, 'roles_names'] = update_data.roles_names
        updates_made.append("roles_names")
    
    if update_data.location is not None:
        sketches_df.at[sketch_index, 'location'] = update_data.location
        updates_made.append("location")
    
    if update_data.lighting is not None:
        sketches_df.at[sketch_index, 'lighting'] = update_data.lighting
        updates_made.append("lighting")
    
    if update_data.languages is not None:
        sketches_df.at[sketch_index, 'languages'] = update_data.languages
        updates_made.append("languages")
    
    if update_data.done is not None:
        sketches_df.at[sketch_index, 'done'] = update_data.done
        updates_made.append("done")
    
    if not updates_made:
        return UpdateResponse(
            success=False,
            message="No fields provided for update"
        )
    
    # Get updated sketch info
    updated_row = sketches_df.iloc[sketch_index]
    updated_sketch = dataframe_row_to_sketch_info(updated_row, sketch_index)
    
    return UpdateResponse(
        success=True,
        message=f"Successfully updated fields: {', '.join(updates_made)}",
        updated_sketch=updated_sketch
    )


@app.get("/sketches/{sketch_index}", response_model=SketchInfo)
async def get_sketch_by_index(
    sketch_index: int,
    df: pd.DataFrame = Depends(get_data_dependency)
):
    """
    Get a specific sketch by its index
    """
    if sketch_index not in df.index:
        raise HTTPException(
            status_code=404,
            detail=f"Sketch with index {sketch_index} not found"
        )
    
    row = df.iloc[sketch_index]
    return dataframe_row_to_sketch_info(row, sketch_index)


@app.get("/stats", response_model=Dict[str, Any])
async def get_statistics(
    df: pd.DataFrame = Depends(get_data_dependency)
):
    """
    Get statistical information about the sketches
    """
    stats = {
        "total_sketches": len(df),
        "completed_sketches": len(df[df['done'] == 1.0]),
        "incomplete_sketches": len(df[df['done'] == 0.0]),
        "unique_locations": df['location'].nunique(),
        "unique_languages": df['languages'].nunique(),
        "most_common_location": df['location'].mode().iloc[0] if not df['location'].mode().empty else None,
        "most_common_language": df['languages'].mode().iloc[0] if not df['languages'].mode().empty else None,
        "average_actors_per_sketch": df['main_actors_count'].mean(),
        "max_actors_in_sketch": df['main_actors_count'].max(),
        "completion_rate": (len(df[df['done'] == 1.0]) / len(df)) * 100
    }
    
    return stats


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    print("🎭 Starting Kargin Haghordum API...")
    print("📚 Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Search endpoint: http://localhost:8000/search?q=your_query")
    print("🎲 Random sketch: http://localhost:8000/random")
    print("❤️ Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "kargin_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
