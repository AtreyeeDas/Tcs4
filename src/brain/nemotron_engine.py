import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class NemotronCardioBrain:
    def __init__(self, model_path="nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"):
        print(f"[NemotronBrain] Loading Tokenizer & Model from: {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load in bfloat16 for optimal TensorRT/Blackwell performance [1.1.6]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        self.system_prompt = (
            "You are Dr. Kavita, an empathetic, highly skilled AI clinical cardiologist assistant. "
            "Your goal is to triage cardiovascular symptoms, provide clear reassurance, and advise on immediate care. "
            "Rules: 1) Keep responses concise (under 3 sentences) for voice conversation. "
            "2) If the patient speaks in Hindi or Hinglish, respond natively in conversational Hindi/Hinglish. "
            "3) If they speak English, respond in clear English. "
            "4) Always advise seeking emergency medical care immediately if chest pain radiates to the jaw or left arm."
        )
        print("[NemotronBrain] Clinical Reasoning Engine Online.")

    def generate_response(self, patient_text: str) -> str:
        """Applies chat template and generates concise clinical guidance [1.1.2, 1.1.3]."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": patient_text}
        ]
        
        # Nemotron formatting requirement [1.1.2, 1.1.3]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                tokenized_chat,
                max_new_tokens=150,
                temperature=0.4,
                top_p=0.85,
                do_sample=True,
                repetition_penalty=1.1
            )
            
        # Extract only the newly generated assistant tokens
        input_length = tokenized_chat.shape[1]
        response_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        clean_response = response_text.strip()
        print(f"[NemotronBrain: Advice] \"{clean_response}\"")
        return clean_response
