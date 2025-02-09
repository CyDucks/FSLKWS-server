from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io
from main import KeywordSpotter
from typing import Dict, List
import json
import logging
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Keyword Spotter API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the keyword spotter
keyword_spotter = KeywordSpotter()

@app.post("/add_keyword")
async def add_keyword(keyword: str, audio_file: UploadFile):
    """
    Add a new keyword template with its audio sample
    """
    try:
        # Read the audio file content
        contents = await audio_file.read()
        
        try:
            # Try to read the audio data
            audio_data, sample_rate = sf.read(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format. Please provide a valid WAV file."
            )

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if necessary
        if sample_rate != keyword_spotter.sample_rate:
            logger.info(f"Resampling audio from {sample_rate}Hz to {keyword_spotter.sample_rate}Hz")
            audio_data = librosa.resample(
                y=audio_data,
                orig_sr=sample_rate,
                target_sr=keyword_spotter.sample_rate
            )
        
        # Add the keyword
        try:
            keyword_spotter.add_keyword(keyword, audio_data)
        except Exception as e:
            logger.error(f"Error adding keyword: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing keyword: {str(e)}"
            )

        return {"message": f"Successfully added keyword: {keyword}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_keyword")
async def detect_keyword(audio_file: UploadFile, threshold: float = 0.85):
    """Detect keywords in the provided audio file"""
    try:
        # Read the audio file content
        contents = await audio_file.read()
        logger.info(f"Received file: {audio_file.filename}, content type: {audio_file.content_type}")
        
        try:
            # Try to read the audio data
            audio_data, sample_rate = sf.read(io.BytesIO(contents))
            logger.info(f"Successfully read audio file: {audio_file.filename}, sample rate: {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format. Please provide a valid WAV file."
            )

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if necessary
        if sample_rate != keyword_spotter.sample_rate:
            logger.info(f"Resampling audio from {sample_rate}Hz to {keyword_spotter.sample_rate}Hz")
            audio_data = librosa.resample(
                y=audio_data,
                orig_sr=sample_rate,
                target_sr=keyword_spotter.sample_rate
            )
        
        # Log registered keywords
        keywords = list(keyword_spotter.keyword_templates.keys())
        logger.info(f"Currently registered keywords: {keywords}")
        
        # Detect keyword
        try:
            keyword, confidence = keyword_spotter.detect_keyword(audio_data, threshold)
            
            result = {
                "detected_keyword": keyword,
                "confidence": float(confidence) if confidence else 0.0,
                "status": "detected" if keyword else "no_match"
            }
            
            logger.info(f"Detection result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting keyword: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/keywords")
async def list_keywords():
    """
    Get a list of all registered keywords
    """
    try:
        return {"keywords": list(keyword_spotter.keyword_templates.keys())}
    except Exception as e:
        logger.error(f"Error listing keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/keywords/{keyword}")
async def delete_keyword(keyword: str):
    """
    Delete a registered keyword
    """
    try:
        if keyword in keyword_spotter.keyword_templates:
            del keyword_spotter.keyword_templates[keyword]
            return {"message": f"Successfully deleted keyword: {keyword}"}
        else:
            raise HTTPException(status_code=404, detail="Keyword not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting keyword: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 