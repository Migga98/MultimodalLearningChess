import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset, random_split
from rtpt import RTPT
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, VisionEncoderDecoderModel
import time
import pandas as pd
import pickle
import numpy as np
import os

from constants import LABELS

class Image_Caption_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.pixel_values = []

        text = df.text.copy()
        states = df.img.copy()
        for txt in text:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>',
                                       truncation=True, max_length=max_length, padding="max_length")
            encodings_dict['input_ids'] = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in
                                           encodings_dict.input_ids]
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

        for state in states:
            self.pixel_values.append(torch.tensor(state))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.pixel_values[idx].squeeze(), self.input_ids[idx], self.attn_masks[idx]

def proc_time(b_sz, model, n_iter=10):
    dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=b_sz,  # Trains with this batch size.
        pin_memory=True,
        num_workers=2,
    )
    for step, batch in enumerate(dataloader):
        pixel_values = batch[0].to(device).float()
        b_labels = batch[1].to(device)
        b_masks = batch[2].to(device)

        break

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        model(pixel_values,
              labels=b_labels,
              decoder_attention_mask=b_masks, )
    torch.cuda.synchronize()
    end = time.time() - start
    throughput = b_sz * n_iter / end
    print(f"Batch: {b_sz} \t {throughput} samples/sec")
    return b_sz, throughput,


if __name__ == "__main__":

    directory = "./data/datasets/principal_variations_preprocessed"
    data = []
    for idx, file in enumerate((os.listdir(directory))):
        if file == ".gitignore":
            continue
        filename = os.path.join(directory, file)
        with open(filename, 'rb') as fo:
            data.extend(pickle.load(fo, encoding='bytes'))
            break

    data = np.array(data, dtype=object)

    state_data = []
    comment_data = []
    length = data.shape[0]
    for i in range(int(length * 0.001)):
        state = data[:][i][0]
        comment = data[:][i][3]
        state_data.append(state)
        comment_data.append(comment)

    df = pd.DataFrame.from_dict({"img": state_data, "text": comment_data}, )
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')
    dataset = Image_Caption_Dataset(df, tokenizer)
    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size


    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    label2id, id2label = dict(), dict()
    for i, label in enumerate(LABELS):
        label2id[label] = i
        id2label[i] = label
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    decoder = GPT2LMHeadModel.from_pretrained("gpt2", is_decoder=True, add_cross_attention=True,
                                              output_hidden_states=False)
    encoder = AutoModel.from_pretrained("Migga/ViT_Chess_DGX_V1", use_auth_token=True, )

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    decoder.resize_token_embeddings(len(tokenizer))
    decoder.config.decoder_start_token_id = tokenizer.bos_token_id
    decoder.config.pad_token_id = tokenizer.pad_token_id
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model=encoder, decoder_model=decoder
    )
    model.config.block_size = 128
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id



    model.to(device)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rtpt = RTPT(name_initials='MP', experiment_name="Test_batch_size_ViT", max_iterations=len(batch_sizes))
    rtpt.start()
    for batch in batch_sizes:
        _, throughput = proc_time(batch, model)
        rtpt.step(subtitle=f"samples/sec={throughput:2.2f}")
