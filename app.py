import os
from PIL import Image
import torch
import torchaudio
from IPython.display import Audio
import google
from google import genai
from huggingface_hub import login
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from utils import load_model, extract_text, generate_audio

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
AUDIO_FOLDER = os.path.join('static', 'audio')
app.config['UPLOAD'] = UPLOAD_FOLDER
app.config['AUDIO'] = AUDIO_FOLDER

## login to HF
# login(token=os.getenv("HUGGING_FACE_TOKEN"))
# api = HfApi()

## loads model
model = load_model()

@app.route("/", methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        
        image = Image.open(img)
        response = extract_text(img=image)
        
        audio = generate_audio(model=model, text=response)

        if AUDIO_FOLDER:
            torchaudio.save(
                os.path.join(app.config['AUDIO'], f"audio_{filename}.wav"),
                # sample_rate=sample_rate
                sample_rate=16000
                )
        else:
            os.chdir('static')
            os.mkdir('audio')
            torchaudio.save(
                os.path.join('audio', f"audio_{filename}.wav"),
                # sample_rate=sample_rate
                sample_rate=16000
                )
            os.chdir('..')
            
        audio = os.path.join(app.config['AUDIO'], f"audio_{filename}.wav")

        return render_template('homepage.html',
                               img=img,
                               response=response, 
                               audio=audio
                               )
    return render_template('homepage.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)