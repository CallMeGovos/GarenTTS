import torch
import soundfile as sf
import librosa
import yaml
from flask import Flask, Response, request
from models import StyleTTS2
from text_utils import TextCleaner
from utils import compute_style

app = Flask(__name__)

def load_model(config_path, checkpoint_path):
    """Load StyleTTS2 model on GPU if available."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = StyleTTS2(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def generate_tts(text, ref_audio_path, model, device, alpha=0.9, beta=0.9, diffusion_steps=5):
    """Generate speech from text using StyleTTS2."""
    textcleaner = TextCleaner()
    text = textcleaner(text)
    
    # Load reference audio and compute style
    ref_audio, _ = librosa.load(ref_audio_path, sr=24000)
    ref_s = compute_style(ref_audio)
    
    # Generate speech on GPU
    with torch.no_grad():
        wav = model.inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps)
    
    # Convert to CPU for saving
    wav = wav.cpu().numpy()
    return wav

# Load model at startup
config_path = "Models/GarenGodKing/config_ft.yml"
checkpoint_path = "Models/GarenGodKing/epoch_2nd_00049.pth"
model, device = load_model(config_path, checkpoint_path)

@app.route('/generate_wav', methods=['POST'])
def generate_wav():
    """API endpoint to generate WAV from text."""
    try:
        text = request.json['text']
        if not text:
            return {"error": "No text provided"}, 400
        
        # Generate audio
        ref_audio_path = "DataGaren/wavs/0060.wav"
        wav = generate_tts(text, ref_audio_path, model, device)
        
        # Save to temporary buffer
        import io
        buffer = io.BytesIO()
        sf.write(wav, buffer, 24000, format='WAV')
        buffer.seek(0)
        
        return Response(buffer.read(), mimetype='audio/wav')
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)