import librosa
import matplotlib.pyplot as plt
import os
import json
import math
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import langdetect
from scipy.io.wavfile import write
import re
from scipy import signal
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, required=False, default="сайн байна уу та")
parser.add_argument('--noise_scale', type=float, default=0.667, help="Controls variation in tone")
parser.add_argument('--noise_scale_w', type=float, default=0.8, help="Controls variation in prosody")
parser.add_argument('--length_scale', type=float, default=1.0, help="Speech speed: lower=faster, higher=slower")
args = parser.parse_args()

# Configuration
path_to_config = "configs/mongolian.json"
path_to_model = "/kaggle/working/MB-iSTFT-VITS2/logs/models/test/G_30000.pth"
input = args.text
noise_scale = args.noise_scale
noise_scale_w = args.noise_scale_w
length_scale = args.length_scale

# Device check
device = "cuda:0" if torch.cuda.is_available() else "cpu"
hps = utils.get_hparams_from_file(path_to_config)

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

# Load model
net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()
_ = utils.load_checkpoint(path_to_model, net_g, None)

# Text processing
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)

# Optional language tag for multilingual
def langdetector(text):
    try:
        lang = langdetect.detect(text).lower()
        if lang == 'ko': return f'[KO]{text}[KO]'
        elif lang == 'ja': return f'[JA]{text}[JA]'
        elif lang == 'en': return f'[EN]{text}[EN]'
        elif lang == 'zh-cn': return f'[ZH]{text}[ZH]'
        else: return text
    except Exception:
        return text

# Output directory
sid = 0
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
speakers = getattr(hps, 'speakers', [])

# Inference (single speaker)
def vcss(inputstr):
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    stn_tst = get_text(fltstr, hps)
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(
            x_tst, x_tst_lengths,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale
        )[0][0, 0].data.cpu().float().numpy()
    write(f'{output_dir}/output_{sid}.wav', hps.data.sampling_rate, audio)
    print(f'{output_dir}/output_{sid}.wav Generated!')

# Inference (multi-speaker)
def vcms(inputstr, sid):
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    stn_tst = get_text(fltstr, hps)
    for idx, speaker in enumerate(speakers):
        sid = torch.LongTensor([idx]).to(device)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            audio = net_g.infer(
                x_tst, x_tst_lengths, sid=sid,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale
            )[0][0, 0].data.cpu().float().numpy()
        write(f'{output_dir}/{speaker}.wav', hps.data.sampling_rate, audio)
        print(f'{output_dir}/{speaker}.wav Generated!')

# Voice conversion (optional)
def ex_voice_conversion(sid_tgt):
    output_dir = 'ex_output'
    dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    loader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=1, pin_memory=False, drop_last=True, collate_fn=collate_fn)
    data_list = list(loader)
    with torch.no_grad():
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.to(device) for x in data_list[0]]
        audio = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0, 0].data.cpu().float().numpy()
    write(f'{output_dir}/output_{sid_src}-{sid_tgt}.wav', hps.data.sampling_rate, audio)
    print(f'{output_dir}/output_{sid_src}-{sid_tgt}.wav Generated!')

# Run single speaker synthesis
vcss(input)
