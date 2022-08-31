import torch
from rtpt import RTPT
from transformers import ViTForImageClassification, ViTConfig

import time

from constants import LABELS


def proc_time(b_sz, model, n_iter=10):
    x = torch.rand(b_sz, 24, 8, 8).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        model(x)
    torch.cuda.synchronize()
    end = time.time() - start
    throughput = b_sz * n_iter / end
    print(f"Batch: {b_sz} \t {throughput} samples/sec")
    return b_sz, throughput,


if __name__ == "__main__":
    label2id, id2label = dict(), dict()
    for i, label in enumerate(LABELS):
        label2id[label] = i
        id2label[i] = label

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    configuration = ViTConfig(num_hidden_layers=6,
                              num_attention_heads=6,
                              image_size=8,
                              patch_size=2,
                              num_channels=24,
                              num_labels=1968,
                              add_pooling_layer=False,
                              hidden_size=1968,
                              intermediate_size=1 * 1968,
                              # layer_norm_eps=0.0,
                              encoder_stride=1,
                              # qkv_bias=False,
                              id2label=id2label,
                              label2id=label2id,
                              # hidden_dropout_prob=0.1,
                              # attention_probs_dropout_prob=0.1
                              )
    model = ViTForImageClassification(configuration)

    model.cuda()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ]
    rtpt = RTPT(name_initials='MP', experiment_name="Test_batch_size_ViT", max_iterations=len(batch_sizes))
    rtpt.start()
    for batch in batch_sizes:
        _, throughput = proc_time(batch, model)
        rtpt.step(subtitle=f"samples/sec={throughput:2.2f}")
