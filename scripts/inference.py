import sys
sys.path.append('/owrkspace/MetaTTS')

import argparse
import os

import json
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from dataset import TTSDataset
from lightning.collate import get_single_collate
from lightning.model import FastSpeech2Loss, FastSpeech2
from lightning.utils import LightningMelGAN

from utils.tools import synth_samples

from dataclasses import dataclass, asdict

from pathlib import Path
from collections import OrderedDict

# synth utils
import g2p_en
from text import text_to_sequence
from scipy.io import wavfile

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

def to_device(x, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, tuple):
        return tuple([to_device(e, device) for e in x])
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x

def synthesize(args):

    # Read Config
    preprocess_configs = [
        yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        for path in args.preprocess_config
    ]
    model_config = yaml.load(
        open(args.model_config, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config[0], "r"), Loader=yaml.FullLoader
    )
    train_config.update(
        yaml.load(open(args.train_config[1], "r"), Loader=yaml.FullLoader)
    )
    algorithm_config = yaml.load(
        open(args.algorithm_config, "r"), Loader=yaml.FullLoader
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocoder = LightningMelGAN()
    model = get_model(args.ckpt_file, preprocess_configs[0], model_config, algorithm_config)
    model.to(device)
    vocoder.mel2wav.to(device)

    synth_input = prepare_inputs(args.text, 0)
    wf = synthesize_one(model, vocoder, synth_input)
    wavfile.write(args.output_path, 22050, wf)
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, nargs='+', help="path to preprocess.yaml",
        default=['scripts/meta-tts/config/preprocess/freeman.yaml'],
    )
    parser.add_argument(
        "-m", "--model_config", type=str, help="path to model.yaml",
        default='scripts/meta-tts/config/model/base.yaml'
    )
    parser.add_argument(
        "-t", "--train_config", type=str, nargs='+', help="path to train.yaml",
        default=['scripts/meta-tts/config/train/base.yaml', 'scripts/meta-tts/config/train/freeman.yaml'],
    )
    parser.add_argument(
        "-a", "--algorithm_config", type=str, help="path to algorithm.yaml",
        default='scripts/meta-tts/config/algorithm/base_emb_vad.yaml'
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to checkpoint file",
        default="models/meta_emb_vad/checkpoint.pth",
    )
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--speaker_idx', type=int, default=0, help="Integer speaker id")
    parser.add_argument(
        '--out_path', type=str, default='sample.wav', help='Path to save output audio file'
    )

    args = parser.parse_args()
    synthesize(args)
