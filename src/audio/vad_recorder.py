import time
import torch
import pyaudio
import numpy as np

class VADRecorder:
    def __init__(self, sample_rate=16000, chunk_size=512, silence_duration=1.2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.format = pyaudio.paInt16
        self.channels = 1
        
        print("[VADRecorder] Loading Silero VAD from TorchHub...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model.eval()
        self.pa = pyaudio.PyAudio()
        print("[VADRecorder] Audio Microphone Ready.")

    def listen_for_speech(self) -> np.ndarray:
        """Blocks and listens to microphone until speech is detected and followed by silence."""
        stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        audio_frames = []
        speech_started = False
        silence_start_time = None
        
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                # Convert 16-bit PCM bytes to FP32 numpy array normalized between -1.0 and 1.0
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check voice activity confidence
                tensor_chunk = torch.from_numpy(audio_chunk).squeeze()
                speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()
                
                if speech_prob > 0.5:
                    if not speech_started:
                        print("⚡ Voice Activity Detected! Recording...")
                        speech_started = True
                    audio_frames.append(audio_chunk)
                    silence_start_time = None
                elif speech_started:
                    audio_frames.append(audio_chunk)
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > self.silence_duration:
                        # End of utterance reached
                        break
        finally:
            stream.stop_stream()
            stream.close()
            
        if not audio_frames:
            return np.array([], dtype=np.float32)
            
        return np.concatenate(audio_frames)
