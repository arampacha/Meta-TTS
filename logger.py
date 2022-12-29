from typing import List


class BaseLogger():

    def log(self, data:dict, step:int):
        raise NotImplementedError

class WandbLogger(BaseLogger):

    def __init__(self, project, run_name) -> None:
        try:
            import wandb
        except:
            raise ImportError("WandB not installed")
        self._wandb = wandb

    def __eneter__(self, **kwargs):
        self.run = self._wandb.init(**kwargs)
        return self.run

    def __exit__(self):
        self.run.finish()

    def log(self, data:dict, step:int):
        self._wandb.log(data, step)

    def log_audios(self, wavs:List, captions:List[str], step:int, sample_rate=None):
        wavs_to_log = {
            f"audio_{i+1}.wav": wandb.Audio(wav, caption=caption, sample_rate=sample_rate) for i, (wav, caption) in enumerate(zip(wavs, captions))
        }
        self._wandb.log(wavs_to_log, step=step)

class TensorBoardLogger(BaseLogger):

    def __init__(self, logdir) -> None:
        super().__init__()
        try:
            import tensorboard
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(logdir)
        except:
            raise ImportError("TensorBoard not installed.")

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.writer.close()

    def log(self, data:dict, step:int):
        for k, v in data.items():
            self.writer.add_scalar(k, v, global_step=step)

    def log_audios(self, wavs, captions, step:int, sample_rate:int=44100, prefix=""):
        for i, wav in enumerate(wavs):
            self.writer.add_audio(f'{prefix}audio{i+1}.wav', wav, global_step=step, sample_rate=sample_rate)
