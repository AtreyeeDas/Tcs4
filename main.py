import os
import sys
import torch
import numpy as np

# --- PYTORCH 2.6+ SECURITY OVERRIDE ---
# Prevents unpickling/safetensor errors when loading complex multi-module weights on PyTorch Nightly
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
# --------------------------------------

from src.audio.vad_recorder import VADRecorder
from src.speech.seamless_engine import SeamlessSpeechHub
from src.brain.nemotron_engine import NemotronCardioBrain

def main():
    print("=================================================================")
    print("   EDGE CARDIO CARE VOICE PIPELINE (Blackwell sm_120 Edition)    ")
    print("=================================================================")
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("[Fatal Error] CUDA not detected! Blackwell GPU is required.")
        sys.exit(1)
    
    print(f"[Hardware] GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"[Hardware] CUDA Version: {torch.version.cuda}")

    # 1. Initialize Engines
    print("\n--- Initializing Modules ---")
    recorder = VADRecorder(sample_rate=16000, silence_duration=1.2)
    speech_hub = SeamlessSpeechHub(model_path="facebook/seamless-m4t-v2-large")
    brain = NemotronCardioBrain(model_path="nvidia/Nemotron-4-Mini-Hindi-4B-Instruct")

    print("\n[System] All engines online. Dr. Kavita is ready to listen.")
    print("[System] Speak into your microphone in English, Hindi, or Hinglish.")
    print("[System] Press Ctrl+C at any time to exit.\n")

    try:
        while True:
            # Step A: Capture speech using Silero VAD
            print("-----------------------------------------------------------------")
            print("🎙️  LISTENING... (Speak now)")
            audio_numpy = recorder.listen_for_speech()
            
            if audio_numpy is None or len(audio_numpy) == 0:
                continue
                
            # Step B: Transcribe audio to text via SeamlessM4T v2 (ASR)
            # Transcribing to English ('eng') provides clean tokens for the LLM while preserving Indic names/terms
            patient_text = speech_hub.transcribe(audio_numpy, target_lang="eng")
            
            if len(patient_text.strip()) < 3:
                print("[Pipeline] Audio snippet too short or unrecognized. Ignoring.")
                continue

            # Step C: Clinical Reasoning via Nemotron 4B (Brain)
            print("🧠 CLINICAL REASONING...")
            ai_response_text = brain.generate_response(patient_text)
            
            # Step D: Dynamic Language Routing & Speech Synthesis (TTS)
            # Detect if response contains Devanagari characters to route voice synthesizer properly
            has_devanagari = any("\u0900" <= c <= "\u097F" for c in ai_response_text)
            tts_lang = "hin" if has_devanagari else "eng"
            
            print(f"🔊 SPEAKING ({tts_lang.upper()})...")
            # speaker_id=15 provides a clear, professional female vocal profile in SeamlessM4T
            speech_hub.synthesize_and_play(ai_response_text, target_lang=tts_lang, speaker_id=15)

    except KeyboardInterrupt:
        print("\n\n[System] Shutting down Edge Cardio Care pipeline. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
