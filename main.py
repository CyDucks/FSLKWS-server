import torch
from backend import util, model
import numpy as np
import pyaudio
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import threading
import time
import noisereduce as nr
from scipy.signal import butter, lfilter
import soundfile as sf
import io
import json
import os
import logging

logger = logging.getLogger(__name__)

class KeywordSpotter:
    def __init__(self, sample_rate=16000, window_size=0.5):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_samples = int(window_size * sample_rate)
        self.audio_buffer = deque(maxlen=self.window_samples)
        
        # Load PLiX model
        try:
            # Load the model using backend.model.load
            self.model = model.load(encoder_name="base", language="multi", device="cpu")
            self.model.eval()  # Set to evaluation mode
            logger.info("✓ PLiX model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Error loading PLiX model: {str(e)}")
            raise
        
        # Store keyword templates
        self.keyword_templates = {}  # Store {keyword: audio_path}
        
        # Add baseline noise classes
        self._add_noise_classes()
        
        self.support_data = None  # Will hold the formatted support set
        
        # Audio recording parameters
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.chunk = 1024
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Add more robust detection parameters
        self.min_duration = 0.2  # Minimum duration in seconds
        self.max_duration = 2.0  # Maximum duration in seconds
        self.energy_threshold = 0.01  # Energy threshold for voice activity
        self.default_threshold = 0.85  # Default similarity threshold
        self.threshold = 0.85  # Default confidence threshold
        self.min_distance_ratio = 1.5  # Minimum ratio between best and second-best match
        
    def _add_noise_classes(self):
        """Add baseline noise classes for better detection"""
        # Generate blank noise (silence)
        blank_noise = np.zeros(self.sample_rate, dtype=np.float32)
        self.keyword_templates["_silence_"] = blank_noise
        
        # Generate white noise
        white_noise = np.random.normal(0, 0.1, self.sample_rate).astype(np.float32)
        self.keyword_templates["_noise_"] = white_noise
        
        logger.info("Added baseline noise classes")
        
    def preprocess_audio(self, audio_data):
        """Preprocess audio with noise reduction and filtering"""
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio_data,
            sr=self.sample_rate,
            prop_decrease=self.noise_reduction_factor
        )
        
        # Apply bandpass filter (300Hz - 3400Hz, typical speech range)
        nyquist = self.sample_rate // 2
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = lfilter(b, a, reduced_noise)
        
        return filtered_audio
    
    def is_speech(self, audio_data):
        """Simple energy-based voice activity detection"""
        energy = np.mean(np.square(audio_data))
        return energy > self.vad_energy_threshold
    
    def extract_features(self, audio_data):
        """Extract enhanced MFCC features from audio data"""
        # Convert to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Apply pre-emphasis
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Extract MFCCs with more coefficients and delta features
        mfccs = librosa.feature.mfcc(
            y=emphasized_audio, 
            sr=self.sample_rate,
            n_mfcc=20,  # Increased from 13
            hop_length=int(0.010 * self.sample_rate),  # 10ms hop length
            n_fft=int(0.025 * self.sample_rate)  # 25ms window
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Add energy feature
        energy = np.mean(np.square(audio_data))
        
        # Combine all features
        combined_features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1),
            np.mean(delta2_mfccs, axis=1),
            [energy]
        ])
        
        return combined_features
    
    def is_valid_audio(self, audio_data):
        """Check if audio segment is valid for keyword detection"""
        duration = len(audio_data) / self.sample_rate
        energy = np.mean(np.square(audio_data))
        
        # Check duration
        if duration < self.min_duration or duration > self.max_duration:
            return False
            
        # Check energy level (voice activity)
        if energy < self.energy_threshold:
            return False
            
        return True
    
    def prepare_support_set(self):
        """Prepare support set from stored templates"""
        if not self.keyword_templates:
            return None
            
        support = {
            "audio": [],
            "labels": [],
            "classes": []
        }
        
        for idx, (keyword, audio_data) in enumerate(self.keyword_templates.items()):
            # Skip noise classes if they're the only ones
            if len(self.keyword_templates) <= 2 and keyword.startswith("_"):
                continue
                
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            
            # Ensure it's the right shape (1 second of audio)
            if len(audio_tensor) > self.sample_rate:
                audio_tensor = audio_tensor[:self.sample_rate]
            elif len(audio_tensor) < self.sample_rate:
                padding = torch.zeros(self.sample_rate - len(audio_tensor))
                audio_tensor = torch.cat([audio_tensor, padding])
            
            # Add batch and channel dimensions
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, sample_rate]
            
            support["audio"].append(audio_tensor)
            support["labels"].append(idx)
            support["classes"].append(keyword)
        
        support["audio"] = torch.cat(support["audio"], dim=0)  # [n_support, 1, sample_rate]
        support["labels"] = torch.tensor(support["labels"])
        
        self.support_data = util.batch_device(support, device="cpu")
        logger.debug(f"Support set prepared with {len(support['classes'])} classes")
        return self.support_data
        
    def add_keyword(self, keyword_name, audio_samples):
        """Add a new keyword template"""
        # Don't allow overwriting noise classes
        if keyword_name.startswith("_"):
            raise ValueError("Keyword names starting with '_' are reserved for system use")
            
        logger.info(f"Adding keyword: {keyword_name}")
        
        # Ensure audio is float32
        audio_samples = audio_samples.astype(np.float32)
        
        # Basic audio validation
        duration = len(audio_samples) / self.sample_rate
        if duration < 0.1 or duration > 2.0:
            raise ValueError(f"Invalid audio duration: {duration:.2f}s (should be between 0.1s and 2.0s)")
        
        # Check energy level
        energy = np.mean(np.square(audio_samples))
        if energy < 1e-6:
            raise ValueError("Audio energy too low - possibly silent audio")
        
        # Store the template
        self.keyword_templates[keyword_name] = audio_samples
        
        # Update support set
        self.prepare_support_set()
        logger.info(f"Successfully added keyword: {keyword_name}")
        
    def detect_keyword(self, audio_data, threshold=None):
        """Detect if any known keyword is present in the audio"""
        if threshold is None:
            threshold = self.threshold
            
        if not self.support_data:
            logger.warning("No keywords registered yet")
            return None, 0.0
            
        try:
            # Prepare query
            audio_data = audio_data.astype(np.float32)
            query_tensor = torch.from_numpy(audio_data).float()
            
            # Ensure it's the right shape
            if len(query_tensor) > self.sample_rate:
                query_tensor = query_tensor[:self.sample_rate]
            elif len(query_tensor) < self.sample_rate:
                padding = torch.zeros(self.sample_rate - len(query_tensor))
                query_tensor = torch.cat([query_tensor, padding])
            
            # Add batch and channel dimensions
            if query_tensor.dim() == 1:
                query_tensor = query_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, sample_rate]
            
            query = {
                "audio": query_tensor
            }
            query = util.batch_device(query, device="cpu")
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(self.support_data, query)
                pred_idx = predictions.item()
                
                # Get predicted class
                predicted_class = self.support_data["classes"][pred_idx]
                
                # Calculate confidence using embeddings
                distances = torch.cdist(
                    query["embeddings"],
                    self.support_data["prototypes"],
                    p=2
                )
                logits = -distances ** 2
                probs = torch.softmax(logits, dim=1)
                confidence = probs[0, pred_idx].item()
                
                logger.debug(f"Predicted class: {predicted_class}")
                logger.debug(f"Confidence: {confidence:.4f}")
                
                if confidence > threshold:
                    logger.info(f"Detected '{predicted_class}' with confidence {confidence:.2f}")
                    return predicted_class, confidence
                else:
                    logger.info(f"No detection: confidence too low ({confidence:.2f} < {threshold})")
            
            return None, confidence
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return None, 0.0
    
    def delete_keyword(self, keyword):
        """Delete a keyword template"""
        # Don't allow deleting noise classes
        if keyword.startswith("_"):
            raise ValueError("Cannot delete system classes")
            
        if keyword in self.keyword_templates:
            del self.keyword_templates[keyword]
            self.prepare_support_set()
            logger.info(f"Deleted keyword: {keyword}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        
        if len(self.audio_buffer) >= self.window_samples:
            # Process the current window
            audio_window = np.array(list(self.audio_buffer))
            
            # Only process if speech is detected
            if self.is_speech(audio_window):
                keyword, confidence = self.detect_keyword(audio_window)
                
                if keyword:
                    print(f"Detected keyword: {keyword} (confidence: {confidence:.2f})")
        
        return (in_data, pyaudio.paContinue)
    
    def start_listening(self):
        """Start listening for keywords"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
    def stop_listening(self):
        """Stop listening for keywords"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()

# Example usage
def main():
    # Initialize the keyword spotter
    spotter = KeywordSpotter()
    
    # Add keyword templates (you would need to provide actual audio samples)
    # Example: spotter.add_keyword("hello", hello_audio_samples)
    
    # Start listening
    spotter.start_listening()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        spotter.stop_listening()

if __name__ == "__main__":
    main()
