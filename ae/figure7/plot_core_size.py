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

core_size_init = pd.read_csv(
    "core_size_results_init.csv", header=None, names=col_names, index_col=0
)
core_size_init.index.astype(str)
core_size_ar = pd.read_csv(
    "core_size_results_ar.csv", header=None, names=col_names, index_col=0
)
core_size_ar.index.astype(str)


df_sorted = core_size_init.sort_index()
areas = df_sorted["area"].tolist()
# print(areas)
# exit()
# areas = [
#     475.52039916931585,
#     826.76355498007,
#     826.76355498007,
#     793.3380639020086,
#     763.3465573533286,
# ]

plt.figure(figsize=(7, 3))

# Create the stacked bar graph
x = 0
for row_index in ["A", "B", "C", "D", "E"]:
    x = x + 1
    values = core_size_init.loc[row_index].tolist()
    bottom = 0
    for i, (category, value) in enumerate(zip(categories, values[2:])):
        if row_index == "A":
            plt.bar(x, value, bottom=bottom, color=colors[i], label=category, width=0.5)
        else:
            plt.bar(x, value, bottom=bottom, color=colors[i], width=0.5)
        bottom += value

plt.ylabel("Latency (s)")
plt.xlabel("Configurations")
plt.xticks([1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()
xticks = plt.gca().get_xticks()
xticklabels = plt.gca().get_xticklabels()
index_to_color_red = list(xticks).index(2)
xticklabels[index_to_color_red].set_color("#76B900")

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(
    [1, 2, 3, 4, 5],
    areas,
    color="dimgray",
    linestyle="dashed",
    marker="x",
    label="Area",
)
ax2.set_ylabel("Area ($mm^2$)")
ax2.set_ylim([0, 1000])
plt.legend(loc="upper right")
plt.savefig(
    "figure7a.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.show()


plt.figure(figsize=(7, 3))
x = 0
for row_index in ["A", "B", "C", "D", "E"]:
    x = x + 1
    values = core_size_ar.loc[row_index].tolist()
    bottom = 0
    for i, (category, value) in enumerate(zip(categories, values[2:])):
        value = value * 1e3
        if row_index == "A":
            plt.bar(x, value, bottom=bottom, color=colors[i], label=category, width=0.5)
        else:
            plt.bar(x, value, bottom=bottom, color=colors[i], width=0.5)
        bottom += value


# Set the title, legend, and display the graph
# plt.title(
#     "Generation latency under different organization"
# )
plt.ylabel("Latency (ms)")
plt.xlabel("Configurations")
plt.xticks([1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.2, 1.05))
plt.tight_layout()
xticks = plt.gca().get_xticks()
xticklabels = plt.gca().get_xticklabels()
index_to_color_red = list(xticks).index(2)
xticklabels[index_to_color_red].set_color("#76B900")
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(
    [1, 2, 3, 4, 5],
    areas,
    color="dimgrey",
    linestyle="dashed",
    marker="x",
    label="Area",
)
ax2.set_ylabel("Area ($mm^2$)")
ax2.set_ylim([0, 1000])
plt.legend(loc="upper left")
plt.savefig(
    "figure7b.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)
