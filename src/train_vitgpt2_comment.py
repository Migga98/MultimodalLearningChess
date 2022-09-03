import numpy as np
import pandas as pd
import os
import torch
import pickle
from torch.utils.data import DataLoader, Dataset, random_split, RandomSampler, SequentialSampler
from rtpt import RTPT
from transformers import get_linear_schedule_with_warmup, AdamW
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import upload_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, VisionEncoderDecoderModel
from constants import LABELS
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))



if __name__ == "__main__":
    experiment_name = 'ViTGPT_Chess_comment_DGX_V1'
    batch_size = 32
    epochs = 2
    learning_rate = 2e-5
    warmup_steps = 1e2
    epsilon = 1e-8
    sample_every = 200

    label2id, id2label = dict(), dict()
    for i, label in enumerate(LABELS):
        label2id[label] = i
        id2label[i] = label

    directory = "./data/datasets/gameknot_preprocessed"
    data = []
    for idx, file in enumerate((os.listdir(directory))):
        if file == ".gitignore":
            continue
        filename = os.path.join(directory, file)
        with open(filename, 'rb') as fo:
            data.extend(pickle.load(fo, encoding='bytes'))

    data = np.array(data, dtype=object)

    state_data = []
    comment_data = []
    length = data.shape[0]
    for i in range(int(length * 0.1)):
        state = data[:][i][0]
        comment = data[:][i][2]
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

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,
        pin_memory=True,
        #num_workers=4
    )

    validation_dataloader = DataLoader(
        val_dataset,  #
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,
        pin_memory=True,
        #num_workers=4
    )

    decoder = GPT2LMHeadModel.from_pretrained("gpt2", is_decoder=True, add_cross_attention=True,
                                              output_hidden_states=False)
    encoder = AutoModel.from_pretrained("Migga/ViTGPT_Chess_pv_DGX_V2", use_auth_token=True, )

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



    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon,
                      weight_decay=0.01,
                      correct_bias=False,
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create RTPT object
    rtpt = RTPT(name_initials='MP', experiment_name="ViTGPT2_Chess", max_iterations=epochs)
    # Start the RTPT tracking
    rtpt.start()

    torch.backends.cudnn.benchmark = True
    #scaler = torch.cuda.amp.GradScaler()
    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            pixel_values = batch[0].to(device).float()
            b_labels = batch[1].to(device)
            b_masks = batch[2].to(device)

            for param in model.parameters():
                param.grad = None
            '''
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values,
                                labels=labels,
                                )

            '''
            outputs = model(pixel_values=pixel_values, labels=b_labels, decoder_attention_mask=b_masks,)
            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                         batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                                    pixel_values,
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,
                                    top_k=50,
                                    max_length = 64,
                                    top_p=0.95,
                                    num_return_sequences=1
                                )

                for i, sample_output in enumerate(sample_outputs):
                    comment = batch[1][i].to(device)
                    comment = [caption if caption != -100 else 50258 for caption in comment]
                    print(tokenizer.decode(comment, skip_special_tokens=True))
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                    break  # too many outputs on DGX

                model.train()

            #scaler.scale(loss).backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #scaler.step(optimizer)
            optimizer.step()
            scheduler.step()
            #scaler.update()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        rtpt.step(subtitle=f"loss={avg_train_loss:2.2f}")
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        correct, total = 0, 0
        accuracy = 0.0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            pixel_values = batch[0].to(device).float()
            b_labels = batch[1].to(device)
            b_masks = batch[2].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=b_labels, decoder_attention_mask=b_masks,)

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time,
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    model.push_to_hub(repo_path_or_name=experiment_name)

    df_stats = pd.DataFrame(data=training_stats)
    os.makedirs('./data/train_stats/ViTGPT2_Comment', exist_ok=True)

    df_stats.to_csv(os.path.join("./data/train_stats/ViTGPT2_Comment", experiment_name + ".csv"))
    df_stats.to_csv(os.path.join("./data/train_stats/ViTGPT2_Comment", experiment_name + ".txt"), index=False, sep=' ', mode='a')
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(1, epochs + 1))
    name = "Loss_vs_Epoch_%s.png" % datetime.datetime.today().strftime("%Y-%m-%d-%H-%M")
    savedir = os.path.join("./data/train_stats/ViTGPT2_Comment", name)
    plt.savefig(savedir)
    # plt.show()
    rep_name = "Migga/" + experiment_name
    upload_file(
        path_or_fileobj=savedir,
        path_in_repo=name,
        repo_id=rep_name,
    )
    upload_file(
        path_or_fileobj=savedir,
        path_in_repo=experiment_name + ".csv",
        repo_id=rep_name,
    )
    '''
    upload_file(
        path_or_fileobj=savedir,
        path_in_repo=experiment_name + ".txt",
        repo_id=rep_name,
    )
    '''


