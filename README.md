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
- Pre-trained model and reference audio file (see [Setup](#setup))

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/CallMeGovos/GarenTTS.git
   cd tts-project