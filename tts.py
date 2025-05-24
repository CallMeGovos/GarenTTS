import torch
import soundfile as sf
import librosa
import yaml
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, Response, request
from munch import Munch
from phonemizer.backend import EspeakBackend
from torch import nn
import torch.nn.functional as F
import torchaudio
from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

app = Flask(__name__)

# Set seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)

# Initialize text cleaner and phonemizer
textcleaner = TextCleaner()
global_phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

# Mel-spectrogram transform
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(audio, device, model):
    mel_tensor = preprocess(audio).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)

def load_model(config_path, checkpoint_path):
    """Load the model and its components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load pretrained ASR, F0, and BERT models
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # Build model
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    model = {key: model[key].eval().to(device) for key in model}

    # Load checkpoint
    params_whole = torch.load(checkpoint_path, map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            print(f'{key} loaded')
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # Remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    # Initialize diffusion sampler
    sampler = DiffusionSampler(
        model['diffusion'].diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    return model, device, sampler

def inference(text, ref_s, model, device, sampler, alpha=0.9, beta=0.9, diffusion_steps=5, embedding_scale=1):
    """Generate speech from text using the model."""
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model['text_encoder'](tokens, input_lengths, text_mask)
        bert_dur = model['bert'](tokens, attention_mask=(~text_mask).int())
        d_en = model['bert_encoder'](bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model['predictor'].text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model['predictor'].lstm(d)
        duration = model['predictor'].duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model['decoder'].type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model['predictor'].F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model['decoder'].type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model['decoder'](asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    return out.squeeze().cpu().numpy()[..., :-50]  # Remove weird pulse at the end

# Load model at startup
config_path = "Models/GarenGodKing/config_ft.yml"
checkpoint_path = "Models/GarenGodKing/epoch_2nd_00049.pth"
model, device, sampler = load_model(config_path, checkpoint_path)

# Compute reference style at startup
ref_audio_path = "DataGaren/wavs/0060.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=24000)
audio, index = librosa.effects.trim(ref_audio, top_db=30)
if librosa.get_samplerate(ref_audio_path) != 24000:
    audio = librosa.resample(audio, orig_sr=librosa.get_samplerate(ref_audio_path), target_sr=24000)
ref_s = compute_style(audio, device, model['style_encoder'])

@app.route('/generate_wav', methods=['POST'])
def generate_wav():
    """API endpoint to generate WAV from text."""
    try:
        text = request.json['text']
        if not text:
            return {"error": "No text provided"}, 400

        # Generate audio
        wav = inference(text, ref_s, model, device, sampler)

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