import os
import torch
import google
import regex as re
from google import genai
# import google.auth
# import google.auth.exceptions
# from google.genai import types
# import vertexai
# from vertexai import generative_models
# from vertexai.generative_models import GenerativeModel
from csm.models import ModelArgs, Model
from csm.generator import Generator, load_csm_1b

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def extract_text(img):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
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
        config = ModelArgs(
                backbone_flavor="llama-1B",
                decoder_flavor="llama-100M",
                text_vocab_size=128256,
                audio_vocab_size=2051,
                audio_num_codebooks=32
            )
        model = Model(config).to(
            device,
            dtype=torch.bfloat16
            )
        state_dict = torch.load('./model_weights/model_bestval.pt', map_location=device)
        model = model.load_state_dict(state_dict)
        return model
    except:
        model = load_csm_1b(device)
        return model

def generate_audio(model, text):
    model = model
    model.eval()
    generator_ = Generator(model)
    sample_rate = generator_.sample_rate
    audio = generator_.generate(text=text, speaker=1, topk=70)
    audio = audio.unsqueeze(0).cpu()
    return audio, sample_rate