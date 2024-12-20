import random
from pathlib import Path

import torchaudio


class LJSpeechDataset:
    def __init__(
        self,
        data_dir=r"/kaggle/input/ljspeech/LJSpeech-1.1/wavs",
        limit=None,
        max_len=8192,
        *args,
        **kwargs
    ):
        data_path = Path(data_dir)
        self.audio_paths = []
        self.max_len = max_len

        for audio_path in data_path.iterdir():
            if audio_path[-4:] == ".wav":
                self.audio_paths.append(audio_path)

        self.audio_paths.sort()

        if limit is not None:
            self.audio_paths = self.audio_paths[:limit]

    def __getitem__(self, ind):
        audio_path = self.audio_paths[ind]
        audio, _ = torchaudio.load(audio_path)
        if self.max_len:
            start = random.randint(0, max(0, audio.shape[-1] - self.max_len))
            audio = audio[:, start : start + self.max_len]
        return {"audio": audio}

    def __len__(self):
        return len(self.audio_paths)
