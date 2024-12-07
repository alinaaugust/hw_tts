import json
import os
import random
import shutil
from math import ceil
from pathlib import Path

import torchaudio
import wget
from torch.nn.functional import pad
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from src.utils.spec_utils import MelSpectrogram, MelSpectrogramConfig

URL_LINKS = {
    "data": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


class LJSpeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index = self._sort_index(self._get_or_load_index(part))
        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, ind):
        audio_path = self.index[ind]["path"]
        audio = self.load_audio(audio_path)

        mel_spec = self.mel_spec(audio.unsqueeze(0)).squeeze(0)
        frame_rate = ceil(
            MelSpectrogramConfig.max_len / MelSpectrogramConfig.hop_length
        )

        if audio.shape[1] >= MelSpectrogramConfig.max_len:
            start = random.randint(0, mel_spec.shape[1] - frame_rate - 1)
            mel_spec = mel_spec[:, start : start + frame_rate]
            audio = audio[
                :,
                start
                * MelSpectrogramConfig.hop_length : (start + frame_rate)
                * MelSpectrogramConfig.hop_length,
            ]
        else:
            mel_spec = pad(mel_spec, (0, frame_rate - mel_spec.shape[1]), "constant")
            audio = pad(
                audio, (0, MelSpectrogramConfig.max_len - audio.shape[1]), "constant"
            )
        return {
            "audio": audio,
            "mel_spec": mel_spec,
        }

    def _load_part(self):
        arch_path = self._data_dir / f"LJSpeech-1.1.tar.bz2"
        print(f"Loading dataset")
        wget.download(URL_LINKS["data"], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        wav_dirs = set()
        for dirpath, _, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(list(wav_dirs), desc=f"Preparing ljspeech dataset: {part}"):
            wav_dir = Path(wav_dir)
            meta_path = list(wav_dir.glob("*.csv"))[0]
            with meta_path.open() as f:
                for line in f:
                    w_id = line.split("|")[0]
                    w_text = " ".join(line.split("|")[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    w_info = torchaudio.info(str(wav_path))
                    length = w_info.num_frames / w_info.sample_rate
                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "text": w_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
