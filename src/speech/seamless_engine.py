import os
import torch
import torchaudio
import subprocess
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model

class SeamlessSpeechHub:
    def __init__(self, model_path="facebook/seamless-m4t-v2-large", output_dir="temp_audio"):
        print(f"[SeamlessHub] Loading Processor from: {model_path}...")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Blackwell sm_120 Safeguard: Load weights to CPU first to prevent CUDA init kernels from crashing
        print("[SeamlessHub] Loading 2.3B weights to CPU memory...")
        self.model = SeamlessM4Tv2Model.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        
        print("[SeamlessHub] Pushing SeamlessM4T v2 to RTX PRO 5000 GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print("[SeamlessHub] Speech Engine Online.")

    def transcribe(self, audio_numpy: np.ndarray, sample_rate=16000, target_lang="eng") -> str:
        """Converts raw 16kHz VAD audio numpy arrays into text tokens."""
        print(f"[SeamlessHub: ASR] Processing speech features for target language: '{target_lang}'...")
        
        audio_inputs = self.processor(
            audios=audio_numpy, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        # Ensure FP16 precision matching
        audio_inputs["input_features"] = audio_inputs["input_features"].to(torch.float16)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs, 
                tgt_lang=target_lang, 
                generate_speech=False
            )
            
        transcription = self.processor.decode(output_tokens[0].cpu().squeeze(), skip_special_tokens=True)
        print(f"[SeamlessHub: ASR] Transcribed: \"{transcription.strip()}\"")
        return transcription.strip()

    def synthesize_and_play(self, text: str, target_lang="eng", speaker_id=15):
        """Synthesizes clinical text into speech waveform and executes direct Linux playback [1.1.2, 1.1.3]."""
        # Strip Markdown formatting that can cause unnatural TTS artifacts
        clean_text = text.replace("*", "").replace("#", "").replace("_", "").strip()
        if not clean_text:
            return

        print(f"[SeamlessHub: TTS] Generating audio waveform ({target_lang})...")
        text_inputs = self.processor(
            text=clean_text, 
            src_lang=target_lang, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            speech_output = self.model.generate(
                **text_inputs, 
                tgt_lang=target_lang, 
                speaker_id=speaker_id
            )[0].cpu().numpy().squeeze()
            
        sample_rate = self.model.config.sampling_rate
        output_path = os.path.join(self.output_dir, "response.wav")
        
        # Save on CPU side to bypass GPU vocoder resampling bugs
        audio_tensor = torch.from_numpy(speech_output).unsqueeze(0)
        torchaudio.save(output_path, audio_tensor, sample_rate=sample_rate)
        
        # Direct ALSA playback via Linux aplay
        subprocess.run(["aplay", "-q", output_path], check=False)
