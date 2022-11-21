import sys
sys.path.extend('/code/MetaTTS')

import os
import io
import json
import functools

import yaml
import torch
from smart_open import open
from google.cloud import storage
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
import requests

from lightning.model import FastSpeech2
from lightning.utils import LightningMelGAN
from mel2wav import load_model

from pathlib import Path
from collections import OrderedDict

# synth utils
import g2p_en
from text import text_to_sequence
from scipy.io import wavfile


MNT_DIR = os.getenv('MNT_DIR', '/gcs')
X_API_KEY = os.getenv('X_API_KEY', 'key123')
BUCKET = os.getenv('BUCKET', 'vm-test-v0')

DOWNLOAD_MODELS = (os.getenv('DOWNLOAD_MODELS', 'false').lower() in ('true', 'y', '1'))

health_route = "/health"
predict_route = "/predict"
output_bucket = os.getenv('OUTPUT_BUCKET', 'vm-test-0')
project_id = os.getenv("PROJECT_ID", 'vm-test-0')

def list_blobs(bucket_name:str, prefix:str, delimiter=None):
    "List all blobs on the bucket `bucket_name` matching `prefix`"
    storage_client = storage.Client(project_id)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    return blobs

def download_blobs(bucket_name:str, prefix:str):
    os.makedirs(prefix, exist_ok=True)
    for blob in list_blobs(bucket_name, prefix):
        if blob.name.endswith('/'):
            continue
        blob.download_to_filename(blob.name)

g2p = g2p_en.G2p()

def prepare_inputs(text, speaker_id=None):
    phs = g2p(text)
    toks = torch.tensor([text_to_sequence("{"+ " ".join(phs) + "}", ['english_cleaners'])])
    speaker = torch.tensor([speaker_id]) if speaker_id else torch.zeros((1, ), dtype=torch.long)
    return (speaker, toks, torch.tensor([toks.size(1)]), toks.size(1))

def synthesize_one(model, vocoder, input):
    with torch.no_grad():
        preds = model(*to_device(input))
    return vocoder.infer(preds[1][0].transpose(0,1), 32768.)[0]

def _fix_key(k):
    if k == 'model.speaker_emb.weight':
        return 'speaker_emb.model.weight'
    else:
        return k[6:]

@functools.lru_cache(maxsize=8)
def get_model(checkpoint_path:str, preprocess_config, model_config, algorithm_config):

    model = FastSpeech2(preprocess_config, model_config, algorithm_config)
    state = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = OrderedDict([(_fix_key(k), v) for k, v in state['state_dict'].items() if k.startswith('model.')])
    speaker_file = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
    n_speaker = len(json.load(open(speaker_file, "r")))
    if model_state_dict['speaker_emb.model.weight'].size(0) != n_speaker:
        model_state_dict['speaker_emb.model.weight'] = model_state_dict['speaker_emb.model.weight'][:n_speaker, :].clone()
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    return model

@functools.lru_cache(maxsize=8)
def get_vocoder(vocoder_path:str):
    vocoder = load_model(os.path.join(vocoder_path, 'model_file.pth'))
    return LightningMelGAN(vocoder)


def to_device(x, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, tuple):
        return tuple([to_device(e, device) for e in x])
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x

def get_content_type(headers:dict) -> str:
    if 'accept' in headers:
        return headers['accept']
    elif 'Accept' in headers:
        return headers['Accept']
    else:
        HTTPException(400, 'Missing "accept" header')

def decode_request(request_body:bytes, content_type:str):
    try:
        result = json.loads(request_body)
    except Exception as e:
        raise HTTPException(400, e)
    return result


async def upload_blob(bucket_name, src_file, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(src_file)
    blob.make_public()
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"

def get_model_files(model_dir):
    model_path = os.path.join(model_dir, 'model_file.pth')
    preprocess_config_path = os.path.join(model_dir, 'preprocess_config.yaml')
    model_config_path = os.path.join(model_dir, 'model_config.yaml')
    algorithm_config_path = os.path.join(model_dir, 'algorithm_config.yaml')
    return model_path, preprocess_config_path, model_config_path, algorithm_config_path

async def synthesize(model_dir, vocoder_dir, uuid, text, speaker_id=None):
    model_path, preprocess_config_path, model_config_path, algorithm_config_path = get_model_files(model_dir)
    # Read Config
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    algorithm_config = yaml.load(open(algorithm_config_path, "r"), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocoder = get_vocoder(vocoder_dir)
    model = get_model(model_path, preprocess_config, model_config, algorithm_config)
    model.to(device)
    vocoder.mel2wav.to(device)

    synth_input = prepare_inputs(text, speaker_id)
    wf = synthesize_one(model, vocoder, synth_input)
    out = io.BytesIO()
    wavfile.write(out, 22050, wf)
    file_url = await upload_blob(output_bucket, out, f'outputs/{uuid}.wav')
    return file_url


api = FastAPI()


@api.get(health_route)
def health():
    return {}


@api.post(predict_route)
async def predict(uuid:str, text:str, params:dict) -> JSONResponse:
    """
    Synthesize speech:

    Inputs:
    - `uuid` - audio task uuid; synthesized audio is saved as `outputs/{uuid}.wav` at output bucket
    - `text` - text input
    - `params` - dictionary containing model specific inference parameters:(`model_dir`, `speaker_id`, ...)
    """
    try:
        if not isinstance(text, str):
            raise HTTPException(422, "text should be a string")
        speaker_id = params.get('speaker_id')
        model_dir = params.get('model_dir')
        vocoder_dir = params.get('vocoder_dir')
        if DOWNLOAD_MODELS:
            if not os.path.exists(model_dir):
                download_blobs(BUCKET, model_dir)
        else:
            os.makedirs(os.path.join(MNT_DIR, model_dir), exist_ok=True)
            os.chdir(os.path.join(MNT_DIR))
        
        file_url = await synthesize(model_dir, vocoder_dir, uuid, text, speaker_id)
    except Exception as e:
        print(e)
        raise HTTPException(400, 'Failed to synthesize speech.')
    return JSONResponse({"file_url":file_url})


@api.get('/listdir')
async def list_mnt_dir(path:str=None):
    if path is None:
        path = MNT_DIR
    return JSONResponse({'files':os.listdir(path)})


@api.get('/download')
async def download_models():
    "Download models locally"
    api_url = os.environ.get('API_URL', 'https://tts-api-v0-iweeluwcja-uc.a.run.app')
    model_query = '/v1/searchmodelinstance?limit=100'
    r = requests.get(api_url+model_query, headers={'x-api-key':X_API_KEY})
    if r.status_code != 200:
        raise HTTPException(500, "Failed to fetch models list")
    models_list = r.json()
    weight_prefixes = [m['inference_parameters'].get('model_dir') for m in models_list if m['inference_parameters']]

    for prefix in weight_prefixes:
        if not os.path.isdir(prefix):
            download_blobs(BUCKET, prefix)
