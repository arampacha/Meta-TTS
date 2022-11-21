import os
import shutil
import pandas as pd

df = pd.read_csv('freeman_v2/transcripts.tsv', sep='\t')

for i, row in df.iterrows():
    os.makedirs(f'raw_data/freeman/{row.split}/{row.speaker.split()[-1]}/', exist_ok=True)
    os.makedirs(f'preprocessed_data/freeman/TextGrid/{row.speaker.split()[-1]}/', exist_ok=True)
    with open(f'raw_data/freeman/{row.split}/{row.speaker.split()[-1]}/{i}.lab', 'w') as f:
        f.write(row.transcript)
    shutil.move(f'freeman_v2/wavs/{i}.wav', f'raw_data/freeman/{row.split}/{row.speaker.split()[-1]}/{i}.wav')
    shutil.move(f'preprocessed_data/freeman/TextGrid/{i}.TextGrid', f'preprocessed_data/freeman/TextGrid/{row.speaker.split()[-1]}/{i}.TextGrid')

