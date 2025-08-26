import os
import torch
import io
from io import BytesIO
import google
import regex as re
from google import genai
import boto3
import sys
sys.path.insert(0, 'csm')

from csm.models import ModelArgs, Model
from csm.generator import Generator, load_csm_1b

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

bucket_name = 'sagemaker-studio-533267313146-gevqreh7pje'
file_key = 'model_bestval.pt'

def extract_text(img):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[img, 
                  "Extract the text that is on the main open page of the image. \
                    DO NOT add any extra text or special characters to your response, such as quotation marks that are not already in the image of concern. \
                    DISREGARD words that are in incomplete sentences or are on the opposite page of relevance.\
                    DO NOT extract footnotes, abbreviations, page or verse numbers, or page headings."
                ],
        # config={
        #     "response_mime_type": "application/json",
        #     # "response_schema": list[Response],        
        #     }
    )

    json_response = response.to_json_dict()
    json_response = str(json_response["candidates"][0]["content"]["parts"][0]["text"])
    json_response = json_response.replace("- ", "")

    return json_response

def load_model():
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name='us-east-1'
        )

        weights = s3_client.download_file(bucket_name, file_key, file_key)
        buffer = BytesIO()
        torch.save(weights, buffer)

        model = Model()
        
        model.load_state_dict(
            state_dict=torch.load(file_key, map_location=device)
            )
        return model
    
    except:
        model = load_csm_1b(device)
        return model

def generate_audio(model, text):
    model = model
    model.eval()
    generator_ = Generator(model)
    sample_rate = generator_.sample_rate
    audio = generator_.generate(text=text, speaker=1, temperature=0.7, topk=70)
    audio = audio.unsqueeze(0).cpu()
    return audio, sample_rate