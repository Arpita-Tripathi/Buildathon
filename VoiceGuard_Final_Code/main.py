from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import uvicorn
import traceback
import sys
import logging
from pythonjsonlogger import jsonlogger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.concurrency import run_in_threadpool

# Add parent directory to path to handle imports if run directly
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_KEY, SUPPORTED_LANGUAGES, SUPPORTED_FORMATS
from utils import decode_audio
from model import classifier

# --- 1. Logging Setup (JSON) ---
logger = logging.getLogger("voice_guard")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# --- 2. Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Voice Guard API")
app.state.limiter = limiter

# Strict formatted exception handler
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "message": f"Rate limit exceeded: {exc.detail}"
        }
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

@app.get("/")
def home():
    return {
        "message": "Voice Guard API is Running!",
        "docs_url": "http://localhost:8000/docs",
        "health": "Active"
    }

# --- Schemas ---
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

    @field_validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{v}' not supported. Must be one of {SUPPORTED_LANGUAGES}")
        return v

    @field_validator('audioFormat')
    def validate_format(cls, v):
        if v.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Format '{v}' not supported. Must be {SUPPORTED_FORMATS}")
        return v.lower()



# --- Endpoints ---

@app.post("/api/voice-detection")
@limiter.limit("10/minute")  # Limit: 10 requests per minute per IP
async def detect_voice(
    request: Request,
    payload: VoiceDetectionRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):
    if x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    try:
        # Decode Audio (Async Threadpool)
        # Run CPU-intensive decoding in a separate thread to keep server responsive
        y, sr = await run_in_threadpool(decode_audio, payload.audioBase64)
        
        # Predict
        classification, confidence, explanation = classifier.predict(y, sr)
        
        return {
            "status": "success",
            "language": payload.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }
        
    except ValueError as ve:
        # Validation or decoding error
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(ve)
            }
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Processing failed"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
