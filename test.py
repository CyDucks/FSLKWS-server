import requests
import os
import pytest
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add requests to requirements.txt
BASE_URL = "http://0.0.0.0:8000"

def setup_test_environment():
    """Create necessary directories and files for testing"""
    try:
        # Create test_audio directory if it doesn't exist
        Path("test_audio").mkdir(exist_ok=True)
        
        # Verify test files exist
        required_files = ["test_audio/hello.wav", "test_audio/test_hello.wav"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise Exception(f"Missing required test files: {', '.join(missing_files)}\n"
                          f"Please provide the test audio files in the test_audio directory.")
            
        logger.info("Test environment setup complete")
    except Exception as e:
        logger.error(f"Error setting up test environment: {str(e)}")
        raise

def test_add_keyword():
    """Test adding a new keyword"""
    audio_path = "test_audio/hello.wav"
    
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio file not found: {audio_path}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"audio_file": ("hello.wav", audio_file, "audio/wav")}
            response = requests.post(
                f"{BASE_URL}/add_keyword",
                params={"keyword": "hello"},
                files=files
            )
        
        assert response.status_code == 200, f"Failed with status {response.status_code}: {response.text}"
        assert "Successfully added keyword" in response.json()["message"]
    except Exception as e:
        logger.error(f"Error in add_keyword test: {str(e)}")
        raise

def test_detect_keyword():
    """Test keyword detection"""
    audio_path = "test_audio/test_hello.wav"
    
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio file not found: {audio_path}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"audio_file": ("test_hello.wav", audio_file, "audio/wav")}
            response = requests.post(
                f"{BASE_URL}/detect_keyword",
                files=files,
                params={"threshold": 0.85}
            )
        
        assert response.status_code == 200, f"Failed with status {response.status_code}: {response.text}"
        result = response.json()
        assert "detected_keyword" in result
        assert "confidence" in result
    except Exception as e:
        logger.error(f"Error in detect_keyword test: {str(e)}")
        raise

def test_list_keywords():
    """Test listing all keywords"""
    response = requests.get(f"{BASE_URL}/keywords")
    
    assert response.status_code == 200
    result = response.json()
    assert "keywords" in result
    assert isinstance(result["keywords"], list)

def test_delete_keyword():
    """Test deleting a keyword"""
    # First, ensure we have a keyword to delete
    test_add_keyword()
    
    # Now try to delete it
    response = requests.delete(f"{BASE_URL}/keywords/hello")
    
    assert response.status_code == 200
    assert "Successfully deleted keyword" in response.json()["message"]

def test_delete_nonexistent_keyword():
    """Test deleting a keyword that doesn't exist"""
    response = requests.delete(f"{BASE_URL}/keywords/nonexistent")
    
    assert response.status_code == 404
    assert "Keyword not found" in response.json()["detail"]

def test_invalid_audio_format():
    """Test adding a keyword with invalid audio format"""
    # Create a dummy file with wrong format
    with open("test_audio/invalid.txt", "w") as f:
        f.write("This is not an audio file")
    
    with open("test_audio/invalid.txt", "rb") as invalid_file:
        files = {"audio_file": ("invalid.txt", invalid_file, "text/plain")}
        response = requests.post(
            f"{BASE_URL}/add_keyword",
            params={"keyword": "invalid"},
            files=files
        )
    
    assert response.status_code == 400  # Changed to 400 since it's an invalid format
    os.remove("test_audio/invalid.txt")

if __name__ == "__main__":
    # Setup test environment
    try:
        setup_test_environment()
    except Exception as e:
        print(f"Failed to setup test environment: {str(e)}")
        exit(1)
    
    # Run all tests
    print("Running tests...")
    
    tests = [
        (test_add_keyword, "Add keyword"),
        (test_detect_keyword, "Detect keyword"),
        (test_list_keywords, "List keywords"),
        (test_delete_keyword, "Delete keyword"),
        (test_delete_nonexistent_keyword, "Delete nonexistent keyword"),
        (test_invalid_audio_format, "Invalid audio format")
    ]
    
    for test_func, test_name in tests:
        try:
            test_func()
            print(f"✓ {test_name} test passed")
        except Exception as e:
            print(f"✗ {test_name} test failed: {str(e)}") 