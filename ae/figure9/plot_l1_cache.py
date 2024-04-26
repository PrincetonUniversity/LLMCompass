import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

categories = [
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
col_names = ["area", "latency"] + categories

colors_matmul = sns.color_palette("flare_r", 6)
colors_normalization = sns.color_palette("summer", 3)
colors_gelu = sns.color_palette("pink", 1)
colors_allreduce = sns.color_palette("Blues_r", 2)
colors = colors_matmul + colors_normalization + colors_gelu + colors_allreduce

results_init = pd.read_csv(
    "l1_cache_results_init.csv", header=None, names=col_names, index_col=0
)
results_init.index.astype(int)
results_ar = pd.read_csv(
    "l1_cache_results_ar.csv", header=None, names=col_names, index_col=0
)
results_ar.index.astype(int)

areas = [
    782.1048032068737,
    794.1065561553206,
    826.76355498007,
    848.4527315580167,
    913.304090096728,
    1064.9121549472263,
]

plt.figure(figsize=(7, 3))

# Create the stacked bar graph
x = 0
x_labels = [64, 128, 192, 256, 512, 1024]
for row_index in x_labels:
    x = x + 1
    values = results_init.loc[row_index].tolist()
    bottom = 0
    for i, (category, value) in enumerate(zip(categories, values[2:])):
        if row_index == x_labels[0]:
            plt.bar(x, value, bottom=bottom, color=colors[i], label=category, width=0.5)
        else:
            plt.bar(x, value, bottom=bottom, color=colors[i], width=0.5)
        bottom += value


# Set the title, legend, and display the graph
# plt.title(
#     "Prefilling Latency per Layer"
# )
plt.ylabel("Latency (s)")
plt.xlabel("Local buffer size (KB)")
plt.xticks([1, 2, 3, 4, 5, 6], x_labels)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.2, 1.03))
plt.tight_layout()
xticks = plt.gca().get_xticks()
xticklabels = plt.gca().get_xticklabels()
index_to_color_red = list(xticks).index(3)
xticklabels[index_to_color_red].set_color("#76B900")

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(
    [1, 2, 3, 4, 5, 6],
    areas,
    color="dimgray",
    linestyle="dashed",
    marker="x",
    label="Area",
)
ax2.set_ylabel("Area ($mm^2$)")
ax2.set_ylim([0, 1200])
plt.legend(loc="upper center")

plt.savefig("figure9a.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
plt.show()


plt.figure(figsize=(7, 3))
x = 0
for row_index in x_labels:
    x = x + 1
    values = results_ar.loc[row_index].tolist()
    bottom = 0
    for i, (category, value) in enumerate(zip(categories, values[2:])):
        value = value * 1e3
        if row_index == x_labels[0]:
            plt.bar(x, value, bottom=bottom, color=colors[i], label=category, width=0.5)
        else:
            plt.bar(x, value, bottom=bottom, color=colors[i], width=0.5)
        bottom += value


# Set the title, legend, and display the graph
# plt.title(
#     "Generation Latency per Layer per Token"
# )
plt.ylabel("Latency (ms)")
plt.xlabel("Local buffer size (KB)")
plt.xticks([1, 2, 3, 4, 5, 6], x_labels)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.2, 1.03))
plt.tight_layout()
xticks = plt.gca().get_xticks()
xticklabels = plt.gca().get_xticklabels()
index_to_color_red = list(xticks).index(3)
xticklabels[index_to_color_red].set_color("#76B900")
ax1 = plt.gca()
ax1.set_ylim([0, 1.2])
ax2 = ax1.twinx()
ax2.plot(
    [1, 2, 3, 4, 5, 6],
    areas,
    color="dimgray",
    linestyle="dashed",
    marker="x",
    label="Area",
)
ax2.set_ylabel("Area ($mm^2$)")
ax2.set_ylim([0, 1200])
plt.legend(loc="upper center")
plt.savefig("figure9b.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
