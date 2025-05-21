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