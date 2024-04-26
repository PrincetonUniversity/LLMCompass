import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gmean

categories = ["bs", "seq_len", "latency"] + [
    "Q_K_V",
    "Q_mul_K",
    "A_mul_V",
    "Wo_proj",
    "W1_proj",
    "W2_proj",
    "Softmax",
    "LayerNorm_MHA",
    "LayerNorm_FFN",
    "GeLU",
    "AllReduce_MHA",
    "AllReduce_FFN",
]

A100 = pd.read_csv("A100.csv", header=None, names=categories)
A100["latency"] = A100["latency"] * 1000
our = pd.read_csv("our.csv", header=None, names=categories)
our["latency"] = our["latency"] * 1000

bs_list = [1, 2, 4, 8, 16, 32]
colors_our = sns.color_palette("Blues", 3)[1:]
colors_a100 = sns.color_palette("summer_r", 2)
our_512 = our[our.seq_len == 512][our["bs"].isin(bs_list)]["latency"].tolist()
our_2048 = our[our.seq_len == 2048][our["bs"].isin(bs_list)]["latency"].tolist()
a100_512 = A100[A100.seq_len == 512][A100["bs"].isin(bs_list)]["latency"].tolist()
a100_2048 = A100[A100.seq_len == 2048][A100["bs"].isin(bs_list)]["latency"].tolist()
avg_speedup = gmean(
    np.concatenate(
        (
            np.array(a100_512) / np.array(our_512),
            np.array(a100_2048) / np.array(our_2048),
        )
    )
)
print(avg_speedup)

plt.figure(figsize=(8, 3.5))

x_pos = 0.25
for bs in bs_list:
    if bs == 1:
        seq_len = 512
        plt.bar(
            x_pos,
            A100[(A100.bs == bs) & (A100.seq_len == seq_len)].latency,
            width=0.5,
            label=f"GA100 (seq_len={seq_len})",
            color=colors_a100[0],
        )
        bars = plt.bar(
            x_pos + 0.5,
            our[(our.bs == bs) & (our.seq_len == seq_len)].latency,
            width=0.5,
            label=f"Latency design (seq_len={seq_len})",
            color=colors_our[0],
        )
        for bar in bars:
            bar.set_hatch("//")  # Add diagonal stripes
        seq_len = 2048
        plt.bar(
            x_pos + 1,
            A100[(A100.bs == bs) & (A100.seq_len == seq_len)].latency,
            width=0.5,
            label=f"GA100 (seq_len={seq_len})",
            color=colors_a100[1],
        )
        bars = plt.bar(
            x_pos + 1.5,
            our[(our.bs == bs) & (our.seq_len == seq_len)].latency,
            width=0.5,
            label=f"Latency design (seq_len={seq_len})",
            color=colors_our[1],
        )
        for bar in bars:
            bar.set_hatch("//")  # Add diagonal stripes
    else:
        seq_len = 512
        plt.bar(
            x_pos,
            A100[(A100.bs == bs) & (A100.seq_len == seq_len)].latency,
            width=0.5,
            color=colors_a100[0],
        )
        bars = plt.bar(
            x_pos + 0.5,
            our[(our.bs == bs) & (our.seq_len == seq_len)].latency,
            width=0.5,
            color=colors_our[0],
        )
        for bar in bars:
            bar.set_hatch("//")  # Add diagonal stripes
        seq_len = 2048
        if bs < 164:
            plt.bar(
                x_pos + 1,
                A100[(A100.bs == bs) & (A100.seq_len == seq_len)].latency,
                width=0.5,
                color=colors_a100[1],
            )
            bars = plt.bar(
                x_pos + 1.5,
                our[(our.bs == bs) & (our.seq_len == seq_len)].latency,
                width=0.5,
                color=colors_our[1],
            )
            for bar in bars:
                bar.set_hatch("//")  # Add diagonal stripes
    x_pos += 3

plt.xticks([1, 4, 7, 10, 13, 16], bs_list)
plt.xlabel("Batch Size")
plt.ylabel("Latency (ms)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.grid(True, axis="y", ls="--", c="0.8")
plt.savefig("figure11.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
plt.show()