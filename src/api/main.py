from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import API_HOST, API_PORT
from src.api.routes import router

app = FastAPI(
    title="Vizcom API",
    description="Fashion-focused visual communication API",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["fashion"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to Vizcom API",
        "status": "active",
        "docs_url": "/docs",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "batch_analyze": "/api/v1/batch-analyze"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 