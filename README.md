# Text-to-Speech with StyleTTS2

This project implements a **Text-to-Speech (TTS)** system using the **StyleTTS2** model, a state-of-the-art TTS model for generating high-quality speech from text. The project includes a web interface built with **Streamlit** to allow users to input text and generate audio output.

## Features
- Converts text to speech using the StyleTTS2 model.
- Customizable voice style with a reference audio file.
- Web interface powered by Streamlit for easy interaction.
- Supports English text with high-quality audio output (24kHz).

## Demo
Check out the live demo on [Streamlit Cloud](https://garentts-d8qzvnzumfrbyyasqqpycw.streamlit.app/).

## Prerequisites
To run this project locally, you need:
- Python 3.10 or higher
- A machine with at least 8GB RAM (GPU recommended for faster inference)
- Dependencies listed in `requirements.txt`

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/CallMeGovos/GarenTTS.git
   cd GarenTTS
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Download model**:
   - Download epoch_2nd_00049.pth from [epoch_2nd_00049.pth](https://drive.google.com/drive/folders/1ez4TktS_nEU7NxJjN4D4aZnImOai6QUT).
   - Place in Models/GarenGodKing/.
4. **Install espeak (for phonemizer)**:
   - Windows: Download and install espeak.
   - Linux/Mac:
   ```bash
   sudo apt-get install espeak  # Linux
   brew install espeak         # Mac
5. **Test**:
   - Open file tts_notebook.ipynb
   - Run All.

## Running the Application
1. **Start the Flask API**:
   - Open a terminal and run:
     ```bash
     python tts.py
     ```
   - The API runs at `http://localhost:5000`. Ensure port 5000 is open in your firewall:
     - Windows: Open **Windows Defender Firewall** > **Advanced Settings** > **Inbound Rules** > Create rule for port 5000 (TCP).

2. **Start the Streamlit app**:
   - Open a second terminal and run:
     ```bash
     streamlit run app.py
     ```
   - This starts a web server at `http://localhost:8501`.

3. **Interact with the app**:
   - Open `http://localhost:8501` in your browser.
   - Navigate to the "Tạo Audio" page.
   - Enter text (e.g., "I am Garen.") in the input field.
   - Click "Tạo Audio" to generate and play the audio.

## Notes
- The Flask API (`tts.py`) must be running before starting the Streamlit app.
- TTS processing is GPU-accelerated for faster inference (typically 1-2 seconds per sentence). CPU fallback is available but slower.
- Ensure `epoch_2nd_00049.pth`, `config_ft.yml`, and `0060.wav` are in the correct directories (`Models/GarenGodKing/` and `DataGaren/wavs/`).
- Large files are hosted on Hugging Face/Google Drive to keep the repository lightweight.

## Project Structure
```
GarenTTS/
├── app.py                   # Streamlit app
├── tts.py                   # Flask API with TTS logic
├── requirements.txt         # Dependencies
├── Models/
│   ├── GarenGodKing/
│   │   ├── config_ft.yml    # Model configuration
│   │   ├── epoch_2nd_00049.pth  # Model weights
├── DataGaren/
│   ├── wavs/
│   │   ├── 0060.wav         # Reference audio
├── Modules/
│   ├── diffusion/
│   │   ├── sampler.py       # Diffusion sampler
├── Utils/
│   ├── ASR/
│   │   ├── models.py        # Contains ASRCNN
│   ├── PLBERT/
│   │   ├── util.py          # BERT utilities
├── models.py                # Model definitions
├── utils.py                 # Utility functions
├── text_utils.py            # Text processing
├── README.md                # This file
```

## Backup Instructions
To prevent data loss (e.g., from `git push -f`):
1. **GitHub**:
   ```bash
   git pull origin main  # Sync before pushing
   git add .
   git commit -m "Backup changes"
   git push origin main
   ```
2. **Hugging Face** (for large files):
   ```bash
   git lfs install
   git clone https://huggingface.co/CallMeGovos/GarenTTS_Models
   cd GarenTTS_Models
   cp path/to/epoch_2nd_00049.pth .
   cp path/to/0060.wav .
   git add .
   git commit -m "Backup model and audio"
   git push
   ```
3. **Google Drive**:
   ```bash
   xcopy G:\Project\GarenTTS G:\GarenTTS_Backup /E /H /C /I
   ```
   Upload `GarenTTS_Backup` to Google Drive.

## Future Improvements
- Optimize TTS for lower latency on GPU.
- Support multiple languages and voices.
- Add voice customization options in the Streamlit app.

## Acknowledgments
- Built using [StyleTTS2](https://github.com/yl4579/StyleTTS2).
- Libraries: `torch`, `librosa`, `phonemizer`, `streamlit`, `flask`.

## Contact
For questions, contact [quockhanh2002bd@gmail.com](quockhanh2002bd@gmail.com) or open an issue on GitHub.