import numpy as np
import soundfile as sf

def generate_test_audio(filename, duration=1.0, sample_rate=16000):
    """Generate a simple test audio file with a sine wave"""
    # Generate a 440 Hz sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    # Ensure the audio data is float32
    audio_data = audio_data.astype(np.float32)
    
    # Save the audio file
    sf.write(filename, audio_data, sample_rate)

if __name__ == "__main__":
    # Create test audio files
    generate_test_audio("test_audio/test_hello.wav")
    print("Test audio files generated successfully!") 