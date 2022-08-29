import numpy as np
import pandas as pd
import os
import torch
import pickle
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset, random_split, RandomSampler, SequentialSampler
from rtpt import RTPT
from transformers import ViTForImageClassification, ViTConfig, get_linear_schedule_with_warmup
from transformers import AdamW
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from utils import board_to_planes, save_as_pickle
from constants import LABELS, MV_LOOKUP

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Image_Dataset(Dataset):
    def __init__(self, df, ):
        self.df = df
        self.pixel_values = []
        self.labels = []
        labels = df.label.copy()
        states = df.img.copy()
        for state in states:
            self.pixel_values.append(torch.as_tensor(state))

        for label in labels:
            self.labels.append(torch.as_tensor(label))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.pixel_values[idx].squeeze(), self.labels[idx]


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

'''
from huggingface_hub import notebook_login
sudo apt -qq install git-lfs
git config --global credential.helper store
notebook_login()
'''

if __name__ == "__main__":
    batch_size = 256
    epochs = 3
    learning_rate = 2e-5
    warmup_steps = 1e2
    epsilon = 1e-8
    sample_every = 200

    label2id, id2label = dict(), dict()
    for i, label in enumerate(LABELS):
        label2id[label] = i
        id2label[i] = label

    directory = "./data/datasets/master_games_preprocessed"
    data = []
    for idx, file in enumerate((os.listdir(directory))):
        if file == ".gitignore":
            continue
        filename = os.path.join(directory, file)
        with open(filename, 'rb') as fo:
            data.extend(pickle.load(fo, encoding='bytes'))

    data = np.array(data, dtype=object)

    state_data = []
    move_data = []
    length = data.shape[0]
    for i in range(int(length * 1)):
        state = data[:][i][0]
        move = data[:][i][2]
        move_nr = MV_LOOKUP[move]

        state_data.append(state)
        move_data.append(move_nr)

    df = pd.DataFrame.from_dict({"img": state_data, "label": move_data},)

    dataset = Image_Dataset(df, )

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,
        pin_memory=True,
        # num_workers=4
    )

    validation_dataloader = DataLoader(
        val_dataset,  #
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,
        pin_memory=True,
        # num_workers=4
    )

    configuration = ViTConfig(num_hidden_layers=6, num_attention_heads=6, image_size=8,
                              patch_size=2, num_channels=24, num_labels=1968,
                              add_pooling_layer=False, hidden_size=1968, intermediate_size=1 * 1968,
                              # layer_norm_eps=0.0,
                              encoder_stride=1,
                              # qkv_bias=False,
                              id2label=id2label,
                              label2id=label2id,
                              )
    model = ViTForImageClassification(configuration)

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
    experiment_name = 'ViT_Chess_DGX_V1'
    rtpt = RTPT(name_initials='MP', experiment_name="ViT_Chess", max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()

    #torch.backends.cudnn.benchmark = True

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
            labels = batch[1].to(device).long()

            for param in model.parameters():
                param.grad = None

            outputs = model(pixel_values=pixel_values,
                            labels=labels,
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                         batch_loss, elapsed))

                '''
                model.eval()

                sample_outputs = model.generate(
                                        pixel_values,
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 200,
                                        top_p=0.95, 
                                        num_return_sequences=1
                                    )
                for i, sample_output in enumerate(sample_outputs):
                      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()
                '''

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

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
            labels = batch[1].to(device).long()

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values,
                                labels=labels,
                                )
                loss = outputs[0]

                correct += torch.sum(torch.argmax(outputs[1], dim=1) == labels).item()
                total += len(batch[0])

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)
        accuracy = 100 * correct / total
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation Accuracy: {0:.2f}".format(accuracy))
        print("  Validation Correct Samples: {:}".format(correct))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'Validation Accuracy': accuracy,
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    df_stats = pd.DataFrame(data=training_stats)

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
    plt.savefig(os.path.join("./data/train_stats/ViT_Chess",
                             "Loss_vs_Epoch_%s.png" % datetime.datetime.today().strftime("%Y-%m-%d-%H:%M")))
    # plt.show()

    model.push_to_hub(repo_path_or_name=experiment_name)
