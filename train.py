import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_weights_file_path, get_config, latest_weights_file_path
from dataset import BilingualDataset
from model import Transformer, build_transformer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split='train',
    )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size]
    )

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len'],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len'],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]
        ).ids
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]
        ).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_trgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_trgt_len,
        config['seq_len'],
        config['seq_len'],
        config['d_model'],
    )
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True
    )

    train_dataloader, val_dataloader, src_tokenizer, trgt_tokenizer = get_ds(
        config
    )
    model = get_model(
        config, src_tokenizer.get_vocab_size(), trgt_tokenizer.get_vocab_size()
    ).to(device)

    # logging
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_Step = 0
    preload = config['preload']
    model_filename = (
        latest_weights_file_path(config)
        if preload == 'latest'
        else get_weights_file_path(config, preload) if preload else None
    )
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f'Processing Epoch {epoch:02d}'
        )
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Tensors are sent to the Transformer (Forward pass)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(
                proj_output.view(-1, trgt_tokenizer.get_vocab_size()),
                label.view(-1),
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # logging
            writer.add_scalar('train loss', loss.item(), global_Step)
            writer.flush()

            # Backprop
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_Step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_Step,
            },
            model_filename,
        )


if __name__ == '__main__':
    config = get_config()
    train_model(config)
