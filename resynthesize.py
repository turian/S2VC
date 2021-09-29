#!/usr/bin/env python3
"""Convert multiple pairs."""

import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import yaml
import torch
import numpy as np
import soundfile as sf
from argparse import ArgumentParser
from tqdm import tqdm

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from data.feature_extract import FeatureExtractor
from data.utils import log_mel_spectrogram
from models import load_pretrained_wav2vec


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("info_path", type=str)
    parser.add_argument("output_dir", type=str, default=".")
    parser.add_argument("-s", "--src_feat_name", default="cpc")
    parser.add_argument("-r", "--ref_feat_name", default="cpc")
    parser.add_argument("-w", "--wav2vec_path",
                        default="checkpoints/wav2vec_small.pt")
    parser.add_argument("-v", "--vocoder_path",
                        default="checkpoints/vocoder.pt")

    parser.add_argument("--sample_rate", type=int, default=16000)

    return vars(parser.parse_args())


def main(
    info_path,
    output_dir,
    src_feat_name,
    ref_feat_name,
    wav2vec_path,
    vocoder_path,
    sample_rate,
    **kwargs,
):
    """Main function."""

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    src_feat_model = FeatureExtractor(src_feat_name, wav2vec_path, device)

    ref_feat_model = FeatureExtractor(ref_feat_name, wav2vec_path, device)

    print(f"[INFO] {src_feat_name} is loaded")

    vocoder = torch.jit.load(vocoder_path).to(device).eval()
    print("[INFO] Vocoder is loaded from", vocoder_path)

    path2wav = partial(load_wav, sample_rate=sample_rate, trim=True)

    with open(info_path) as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)

    extractor = partial(
                log_mel_spectrogram,
                preemph=0.97,
                sample_rate=16000,
                n_mels=80,
                n_fft=400,
                hop_length=320,
                win_length=400,
                f_min=0,
                center=False,
            )

    out_mels = []
    with Pool(cpu_count()) as pool:
        for pair_name, pair in tqdm(infos.items()):
            src_wav = load_wav(pair["source"], sample_rate, trim=True)
            #src_wav = torch.FloatTensor(src_wav).to(device)

            out_mel = extractor(src_wav)
            #out_mel = torch.FloatTensor(out_mel).to(device).transpose(0, 1).unsqueeze(0)
            out_mel = torch.FloatTensor(out_mel).to(device)
            print(out_mel.shape)
            out_mels.append(out_mel)
            """
            with torch.no_grad():
                src_mel = (ref_feat_model.get_feature([src_wav])[0].transpose(
                        0, 1).unsqueeze(0))
                #out_mel = out_mel.transpose(1, 2).squeeze(0)
                out_mel = src_mel
                out_mels.append(out_mel)
            """

        # print(f"[INFO] Pair {pair_name} converted")
    # out_mel: batch_size, time_stamp, mel_dim
    del src_feat_model
    del ref_feat_model
    print("[INFO] Generating waveforms...")
    batch_size = 10
    total = len(out_mels)
    out_wavs = []
    pbar = tqdm(total=len(out_mels), ncols=0, unit="wavs")
    with torch.no_grad():
        for i in range(0, total, batch_size):
            out_wavs.extend(vocoder.generate(out_mels[i:i+batch_size]))
            pbar.update(min(batch_size, total))
        if total % batch_size != 0 and total > batch_size:
            out_wavs.extend(vocoder.generate(
                out_mels[total - total % batch_size:]))
            pbar.update(total % batch_size)
    pbar.close()

    print("[INFO] Waveforms generated")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, out_mel, out_wav in tqdm(zip(
        infos.keys(), out_mels, out_wavs
    )):
        out_wav = out_wav.cpu().numpy()
        out_path = Path(out_dir, pair_name)

        plot_mel(out_mel, filename=out_path.with_suffix(".mel.png"))
        sf.write(out_path.with_suffix(".wav"), out_wav, sample_rate)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
