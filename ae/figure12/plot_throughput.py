import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

our_directory = "our/our"
A100_directory = "A100/A100"

categories = ["bs", "s", "latency"] + [
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

throughput_our = []
bs_our = []
latency_our = []
throughput_A100 = []
bs_A100 = []
latency_A100 = []


def get_total_decoding_latency(df: pd.DataFrame, start, end):
    df_filtered = df[(df["s"] >= start) & (df["s"] <= end)]
    total_latency = 0

    # Calculate the mean of the values for each length interval and add to the sum
    for i in range(len(df_filtered) - 1):
        # Calculate the mean of current and next value
        mean = (df_filtered.iloc[i]["latency"] + df_filtered.iloc[i + 1]["latency"]) / 2
        # Calculate the difference in length
        length_interval = df_filtered.iloc[i + 1]["s"] - df_filtered.iloc[i]["s"]
        # Multiply the mean value by the length interval and add to the sum
        total_latency += mean * length_interval

    # print(total_latency)
    return total_latency


for input_length in [256, 512, 1024, 2048]:
    temp_our = []
    temp_A100 = []
    temp_our_bs = []
    temp_A100_bs = []
    temp_our_latency = []
    temp_A100_latency = []
    for output_length in [256, 512, 768, 1024, 1280, 1536, 1792, 2048]:
        our_prefill_df = pd.read_csv(
            f"{our_directory}_{input_length}_{output_length}_prefill.csv",
            header=None,
            names=categories,
        )
        # print(our_prefill_df)
        our_prefill_latency = our_prefill_df.iloc[0]["latency"]
        our_bs = our_prefill_df.iloc[0]["bs"]
        temp_our_bs.append(our_bs)
        our_decoding_df = pd.read_csv(
            f"{our_directory}_{input_length}_{output_length}_decoding.csv",
            header=None,
            names=categories,
        ).sort_values(by="s")
        our_decoding_latency = get_total_decoding_latency(
            our_decoding_df, input_length, input_length + output_length
        )
        # print(our_decoding_latency)
        our_throughput = (
            our_bs * output_length / (our_prefill_latency + our_decoding_latency) / 12
        )
        temp_our.append(our_throughput)
        temp_our_latency.append(our_prefill_latency + our_decoding_latency)

        A100_prefill_df = pd.read_csv(
            f"{A100_directory}_{input_length}_{output_length}_prefill.csv",
            header=None,
            names=categories,
        )
        A100_prefill_latency = A100_prefill_df.iloc[0]["latency"]
        A100_bs = A100_prefill_df.iloc[0]["bs"]
        temp_A100_bs.append(A100_bs)
        A100_decoding_df = pd.read_csv(
            f"{A100_directory}_{input_length}_{output_length}_decoding.csv",
            header=None,
            names=categories,
        ).sort_values(by="s")
        A100_decoding_latency = get_total_decoding_latency(
            A100_decoding_df, input_length, input_length + output_length
        )
        A100_throughput = (
            A100_bs
            * output_length
            / (A100_prefill_latency + A100_decoding_latency)
            / 12
        )
        temp_A100.append(A100_throughput)
        temp_A100_latency.append(A100_prefill_latency + A100_decoding_latency)

    throughput_our.append(temp_our)
    throughput_A100.append(temp_A100)
    bs_our.append(temp_our_bs)
    bs_A100.append(temp_A100_bs)
    latency_our.append(temp_our_latency)
    latency_A100.append(temp_A100_latency)
# print(throughput_our)
# print(throughput_A100)
print(latency_our)
print(latency_A100)
print(
    statistics.geometric_mean(
        (np.array(latency_our) / np.array(latency_A100)).flatten()
    )
)


# Function to convert RGB to grayscale intensity
def get_intensity(color):
    return color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114


cmap = sns.color_palette("viridis", as_cmap=True)
data = np.array(throughput_our)  # / np.array(throughput_A100)
print(data.mean())
fig, ax = plt.subplots()
cax = ax.imshow(data, interpolation="nearest", cmap=cmap)
# cax = sns.heatmap(data, cmap="Blues")

# Add a colorbar
fig.colorbar(cax, shrink=0.5)

# Set a threshold for deciding text color
intensity_threshold = 0.5
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # Get the color from the colormap
        cell_color = cax.cmap(cax.norm(data[i, j]))
        # Calculate intensity of the cell color
        intensity = get_intensity(cell_color)
        # Choose text color based on intensity
        text_color = "white" if intensity < intensity_threshold else "black"
        text = ax.text(
            j, i, int(data[i, j]), ha="center", va="center", color=text_color
        )

# Set the x-axis and y-axis values
x_axis_labels = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
y_axis_labels = [256, 512, 1024, 2048]

# Set ticks positions
ax.set_xticks(np.arange(len(x_axis_labels)))
ax.set_yticks(np.arange(len(y_axis_labels)))

# Set ticks labels
ax.set_xticklabels(x_axis_labels)
ax.set_yticklabels(y_axis_labels)

# Set labels for axes
ax.set_xlabel("Output Length")
ax.set_ylabel("Input Length")
ax.invert_yaxis()
# # Rotate the tick labels for the x-axis if needed
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Show the plot
plt.tight_layout()
plt.savefig("figure12a.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)


import statistics

cmap = sns.color_palette("viridis", as_cmap=True)
data = np.array(throughput_our) / np.array(throughput_A100)
print(statistics.geometric_mean(data.flatten()))
fig, ax = plt.subplots()
cax = ax.imshow(
    data,
    interpolation="nearest",
    cmap=cmap,
)
# cax = sns.heatmap(data, cmap="viridis")

# Add a colorbar
fig.colorbar(cax, shrink=0.5)

# Set a threshold for deciding text color
intensity_threshold = 0.5
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # Get the color from the colormap
        cell_color = cax.cmap(cax.norm(data[i, j]))
        # Calculate intensity of the cell color
        intensity = get_intensity(cell_color)
        # Choose text color based on intensity
        text_color = "white" if intensity < intensity_threshold else "black"
        text = ax.text(
            j, i, round(data[i, j], 2), ha="center", va="center", color=text_color
        )

# Set the x-axis and y-axis values
x_axis_labels = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
y_axis_labels = [256, 512, 1024, 2048]

# Set ticks positions
ax.set_xticks(np.arange(len(x_axis_labels)))
ax.set_yticks(np.arange(len(y_axis_labels)))

# Set ticks labels
ax.set_xticklabels(x_axis_labels)
ax.set_yticklabels(y_axis_labels)

# Set labels for axes
ax.set_xlabel("Output Length")
ax.set_ylabel("Input Length")
ax.invert_yaxis()
# # Rotate the tick labels for the x-axis if needed
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Show the plot
plt.tight_layout()
plt.savefig("figure12b.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
