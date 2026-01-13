from transformers import AutoModelForSeq2SeqLM
import torch
import time
from tqdm import tqdm


def check_encoder_scaling(encoder, k, num_trials=10):
    for i in range(num_trials):
        src = {
            "input_ids": torch.randint(low=0, high=256000, size=(32, k)).to(
                encoder.device
            ),
            "attention_mask": torch.ones(32, k).to(encoder.device),
        }
        src_enc = encoder(**src).last_hidden_state


if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    torch.cuda.set_device(0)
    model.cuda()
    encoder = model.model.encoder
    encoder.train()
    times = dict()
    for k in tqdm(range(5, 100, 1)):
        start = time.time()
        check_encoder_scaling(encoder, k)
        duration = time.time() - start
        times[k] = duration
    for k in times:
        print(f"{k}: {times[k]:.3f}")
    xs = sorted(times.keys())
    ys = [times[k] for k in xs]
    import matplotlib.pyplot as plt

    plt.plot(xs, ys)
    plt.savefig("perf.png")
