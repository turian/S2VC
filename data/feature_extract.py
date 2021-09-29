import torch
import torchcrepe

from functools import partial
from multiprocessing import Pool, cpu_count
from models import load_pretrained_wav2vec
from data import log_mel_spectrogram


class FeatureExtractor:
    def __init__(self, feature_name, wav2vec2_path=None, device=None):
        self.device = device
        if (
            feature_name == "apc"
            or feature_name == "cpc"
            or feature_name == "timit_posteriorgram"
            or feature_name == "fbank"
        ):
            self.extractor = (
                torch.hub.load(
                    "s3prl/s3prl:f2114342ff9e813e18a580fa41418aee9925414e",
                    feature_name,
                    refresh=True,
                )
                .eval()
                .to(device)
            )
            self.mode = 1
        elif feature_name == "wav2vec2":
            self.extractor = load_pretrained_wav2vec(wav2vec2_path).eval().to(device)
            self.mode = 2
        elif feature_name == "wav2vec2_mel":
            self.extractor = partial(
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
            self.mode = 3
        elif feature_name == "cpc_mel":
            self.extractor = partial(
                log_mel_spectrogram,
                preemph=0.97,
                sample_rate=16000,
                n_mels=80,
                n_fft=465,
                hop_length=160,
                win_length=465,
                f_min=80,
                center=True,
            )
            self.mode = 3
        elif feature_name == "crepe":
            torchcrepe.load.model(device=device, capacity="full")
            self.extractor = partial(_torchcrepe, device=device)
            self.mode = 3
        else:
            print(feature_name)
            print(
                "Please use timit_posteriorgram, apc, wav2vec2, cpc, wav2vec2_mel, cpc_mel, crepe, or fbank"
            )
            exit()

    def get_feature(self, wavs):
        if self.mode == 1:
            return self.extractor(wavs)
        elif self.mode == 2:
            feats = []
            for wav in wavs:
                feat = self.extractor.extract_features(wav.unsqueeze(0), None)[
                    0
                ].squeeze(0)
                feats.append(feat)
        elif self.mode == 3:
            wavs = [wav.cpu().numpy() for wav in wavs]
            feats = [self.extractor(wav) for wav in wavs]
            feats = [torch.FloatTensor(feat).to(self.device) for feat in feats]
            return feats

        return feats


def _torchcrepe(x, device):
    embedding = torchcrepe.embed(
        # TODO: Faster if we don't move back and forth from CPU
        audio=torch.tensor(x, device=device).view(1, -1),
        sample_rate=16000,
        # NOTE: This is CPC mel, not wav2vec2 mel
        hop_length=160,
        model="full",
        device=device,
        pad=False,
        # pad=True,
        batch_size=512,
    )
    # Convert 1 x frames x 32x64 embedding to frames x 32*64
    assert embedding.shape[0] == 1
    assert embedding.ndim == 4
    embedding = embedding.view((embedding.shape[1], -1))
    return embedding.to("cpu")
