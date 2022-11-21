import sys
sys.path.append('/workspace/MetaTTS')

import argparse
import os

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

import wandb

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

texts = [
    "This is a voicemod T T Speech trial of zero sample.",
    'This is a test run to understand the voice quality of the audio.'
]

synth_inputs = [prepare_inputs(t) for t in texts]

# train utils
_lnames = ('tot', 'mel', 'postnet_mel', 'pitch', 'energy', 'dur')
def loss2dict(losses, eval=False):
    pref = 'eval' if eval else 'train'
    return {f"{pref}/{n}_loss":v.item() for n, v in zip(_lnames, losses)}

class AverageMeter:
    
    def __init__(self, store_vals=False, store_avgs=False):
        self.store_vals = store_vals
        self.store_avgs = store_avgs
        if store_vals: self.values = []
        if store_avgs: self.avgs = []
        self.sum, self.n, self.avg = 0, 0, 0
        
    def update(self, v, n:int=1):
        if self.store_vals: self.values.append(v)
        self.n += n
        self.avg += (v - self.avg)/self.n
        
    def reset(self):
        if self.store_avgs and self.avg: self.avgs.append(self.avg)
        self.sum, self.n, self.avg = 0, 0, 0

def update_metrics(metrics, values, bs):
    for m, v in zip(metrics, values):
        m.update(v.item()*bs, bs)

def reset_metrics(metrics):
    for m in metrics: m.reset()

@dataclass
class TrainingArguments:

    max_steps:int=1000
    eval_step:int=100
    save_step:int=100
    lr:float=1e-3
    bs:int=48
    grad_acc_step:int=1

    output_dir:str='test-model'


def train(args, configs, train_loader, eval_loader, model, loss_func, optimizer, scheduler, synth_batch, vocoder):
    preprocess_config, model_config, algorithm_config = configs
    eval_metrics = [AverageMeter(store_avgs=True) for _ in _lnames]

    step = 1 #optimization step
    _step = 0 #dataloader step
    pbar = trange(args.max_steps)
    grad_acc_step = args.grad_acc_step

    model.train()
    while True:

        for batch in train_loader:
            _step +=1
            batch = to_device(batch)
            outputs = model(*batch[2:])
            losses = loss_func(batch, outputs)
            loss = losses[0]
            loss /= grad_acc_step
            loss.backward()
            if (_step+1) % grad_acc_step == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                pbar.update()
                
                _log = loss2dict(losses)
                _log['train/lr'] = optimizer.param_groups[0]['lr']
                wandb.log(_log, step=step)
            

            pbar.set_description(f"Train loss: {loss.item()*grad_acc_step:.4f}")
        
            if _step%(args.eval_step*grad_acc_step) == 0:
                model.eval()
                eval_pbar = tqdm(eval_loader, leave=False)
                for batch in eval_pbar:
                    batch = to_device(batch)
                    with torch.no_grad():
                        outputs = model(*batch[2:])
                    eval_losses = loss_func(batch, outputs)
                    update_metrics(eval_metrics, eval_losses, len(batch[0]))
                    eval_pbar.set_description(f"Step {step}. Eval loss: {eval_losses[0].item():.4f}")
                wandb.log({f'eval/{k}_loss':m.avg for k, m in zip(_lnames, eval_metrics)}, step=step)
                reset_metrics(eval_metrics)
                model.train()

            if _step%(args.save_step*grad_acc_step) == 0:
                model.eval()
                #save checkpoint and samples
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'checkpoint_{step-1}.pth'))

                samples_dir = os.path.join(args.output_dir, 'samples', f'{step-1}')
                os.makedirs(samples_dir, exist_ok=True)
                with torch.no_grad():
                    predictions = model(*synth_batch[2:6])
                synth_samples(synth_batch, predictions, vocoder, model_config, preprocess_config, samples_dir)
                wandb.log({n:wandb.Audio(os.path.join(samples_dir, f'{n}.wav'), caption=n) for n in synth_batch[0][:4]}, step=step)

                # synthesize for texts
                wavs_to_log = {}
                for i, synth_input in enumerate(synth_inputs):
                    wf = synthesize_one(model, vocoder, synth_input)
                    fn = os.path.join(samples_dir, f"phrase{i+1}.wav")
                    wavfile.write(fn, 22050, wf)
                    wavs_to_log[f"phrase{i+1}.wav"] = wandb.Audio(fn, caption=texts[i])
                wandb.log(wavs_to_log, step=step)
                model.train()
            
            if step > args.max_steps:
                pbar.close()
                return eval_metrics

def get_model(checkpoint_path:str, preprocess_config, model_config, algorithm_config, spk_emb_init="pretrained", ft_modules=None):

    model = FastSpeech2(preprocess_config, model_config, algorithm_config)

    state = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = OrderedDict([(k[6:], v) for k, v in state['state_dict'].items() if k.startswith('model.')])
    model.load_state_dict(model_state_dict, strict=False)
    #ToDo: report missing/unexpected keys
    # init speaker embeddings
    if spk_emb_init == 'avg':
        model.speaker_emb.model.weight.data[0,:] = model_state_dict['speaker_emb.weight'][:100].mean(dim=0)
    elif spk_emb_init == 'pretrained':
        # src_spk = 0
        _emb_max_idx = model_state_dict['speaker_emb.weight'].size(0)-1
        for i in range(model.speaker_emb.model.weight.size(0)):
            model.speaker_emb.model.weight.data[i,:] = model_state_dict['speaker_emb.weight'].data[min(i, _emb_max_idx),:]
    # select layers to finetune
    # defaults to emb-vad setting
    if ft_modules is None:
        ft_modules = [
            'speaker_emb',
            'variance_adaptor',
            'decoder',
            'mel_linear',
            'postnet'
        ]

    for n, p in model.named_parameters():
        if n.split('.')[0] not in ft_modules:
            p.requires_grad = False
    model.eval()
    return model

def get_num_params(model, verbouse=False):
    total_params = 0
    ft_params = 0
    verbouse = False
    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            ft_params += p.numel()
            if verbouse:print(n)

    print(f"Total parameters: {total_params}; fine-tuned parameters {ft_params}")
    return total_params, ft_params

def to_device(x, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x, tuple):
        return tuple([to_device(e, device) for e in x])
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocoder = LightningMelGAN()

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

    training_args = TrainingArguments(output_dir=args.output_dir)
    configs = (preprocess_configs[0], model_config, algorithm_config)

    model = get_model(args.ckpt_file, preprocess_configs[0], model_config, algorithm_config)

    total_params = 0
    ft_params = 0
    verbouse = False
    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            ft_params += p.numel()
            if verbouse:print(n)
    print(f"Total parameters: {total_params}; fine-tuned parameters {ft_params}")

    train_dataset = TTSDataset(args.train_file, preprocess_configs[0], train_config)
    valid_dataset = TTSDataset(args.validation_file, preprocess_configs[0], train_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=get_single_collate(False),
    )

    eval_loader = DataLoader(
        valid_dataset,
        batch_size=24,
        shuffle=False,
        num_workers=0,
        collate_fn=get_single_collate(False),
    )

    synth_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=get_single_collate(False),
    )

    model.to(device)
    vocoder.mel2wav.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 1e-3, betas=train_config['optimizer']['betas'], eps=train_config['optimizer']['eps'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=training_args.max_steps, pct_start=0.1)

    loss_func = FastSpeech2Loss(preprocess_configs[0], model_config)

    synth_batch = to_device(next(iter(synth_loader)))

    preprocess_config= preprocess_configs[0]

    config = dict(
        dataset=preprocess_config['dataset'],
        arch="fastspeech2",
        task='fine-tune',
        ft_modules = ['speaker_emb','variance_adaptor','decoder','mel_linear','postnet'],
        from_checkpoint = Path(args.ckpt_file).parent.name
    )

    config.update(asdict(training_args))
    config['num_speakers'] = model.speaker_emb.model.weight.size(0)

    with wandb.init(project=args.wandb_project, name=args.wandb_name, tags=['meta-tts'], config=config) as run:

        #log 0 step samples
        samples_dir = os.path.join(args.output_dir, 'samples', '0')
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        wavs_to_log = {}
        for i, synth_input in enumerate(synth_inputs):
            wf = synthesize_one(model, vocoder, synth_input)
            fn = os.path.join(samples_dir, f"phrase{i+1}.wav")
            wavfile.write(fn, 22050, wf)
            wavs_to_log[f"phrase{i+1}.wav"] = wandb.Audio(fn, caption=texts[i])
        wandb.log(wavs_to_log, step=0)

        log = train(training_args, configs, train_loader, eval_loader, model, loss_func, optimizer, scheduler, synth_batch, vocoder)

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
        "-c", "--ckpt_file", type=str, help="ckpt file name",
        default="models/meta-tts/meta_emb_vad/epoch=99-step=99999.ckpt",
    )
    parser.add_argument(
        "--train_file", type=str, help='path to training file'
    )
    parser.add_argument(
        "--validation_file", type=str, help='path to validation file'
    )
    parser.add_argument(
        '--output_dir', type=str, default='models/tmp', help="Path to save model checkpoint"
    )
    parser.add_argument(
        '--wandb_name', type=str, default='test-run',
    )
    parser.add_argument(
        '--wandb_project', type=str, default="VM"
    )

    args = parser.parse_args()
    main(args)
