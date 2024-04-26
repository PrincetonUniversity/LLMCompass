import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


our_decoding = pd.read_csv(
    "our_decoding.csv", header=None, names=["bs", "s", "latency"]
).sort_values(by="s")
our_prefill = pd.read_csv("our_prefill.csv", header=None, names=["bs", "s", "latency"])
A100_decoding = pd.read_csv(
    "A100_decoding.csv", header=None, names=["bs", "s", "latency"]
).sort_values(by="s")
A100_prefill = pd.read_csv(
    "A100_prefill.csv", header=None, names=["bs", "s", "latency"]
)


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


norm_perf = []
for input_length in [256, 512, 1024, 2048]:
    temp_list = []
    our_prefill_latency = our_prefill[our_prefill["s"] == input_length][
        "latency"
    ].values[0]
    A100_prefill_latency = A100_prefill[A100_prefill["s"] == input_length][
        "latency"
    ].values[0]
    for output_length in [256, 512, 768, 1024, 1280, 1536, 1792, 2048]:
        our_total_latency = our_prefill_latency + get_total_decoding_latency(
            our_decoding, input_length, input_length + output_length
        )
        A100_total_latency = A100_prefill_latency + get_total_decoding_latency(
            A100_decoding, input_length, input_length + output_length
        )
        temp_list.append(A100_total_latency / our_total_latency)
    norm_perf.append(temp_list)

cmap = sns.color_palette("viridis", as_cmap=True)
data = np.array(norm_perf)
import statistics

print(statistics.geometric_mean(data.flatten()))
fig, ax = plt.subplots()
cax = ax.imshow(data, interpolation="nearest", cmap=cmap, vmin=0.8, vmax=1)
# cax = sns.heatmap(data, cmap="viridis")

# Add a colorbar
fig.colorbar(cax, shrink=0.5)


# Function to convert RGB to grayscale intensity
def get_intensity(color):
    return color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114


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
plt.savefig("figure10.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)


# norm_perf = []
# norm_perf_ttft = []
# for input_length in [256, 512, 1024, 2048]:
#     temp_list = []
#     our_prefill_latency = our_prefill[our_prefill["s"] == input_length][
#         "latency"
#     ].values[0]
#     A100_prefill_latency = A100_prefill[A100_prefill["s"] == input_length][
#         "latency"
#     ].values[0]
#     for output_length in [256, 512, 768, 1024, 1280, 1536, 1792, 2048]:
#         our_tbt_latency = get_total_decoding_latency(
#             our_decoding, input_length, input_length + output_length
#         )
#         A100_tbt_latency = get_total_decoding_latency(
#             A100_decoding, input_length, input_length + output_length
#         )
#         temp_list.append(our_tbt_latency / A100_tbt_latency)
#     norm_perf.append(temp_list)
#     norm_perf_ttft.append(our_prefill_latency / A100_prefill_latency)

# cmap = sns.color_palette("viridis", as_cmap=True)
# data = np.array(norm_perf)
# data_ttft = np.array(norm_perf_ttft)
# print(data)
# print(data_ttft)
# import statistics
# from matplotlib import gridspec

# print(statistics.geometric_mean(data.flatten()))
# print(statistics.geometric_mean(data_ttft))
# # fig, axs = plt.subplots(1, 2, figsize=(8, 4),
# # 			gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
# fig = plt.figure(figsize=(8, 3))  # Define the figure size
# gs = gridspec.GridSpec(
#     1, 2, width_ratios=[4, 1]
# )  # 2 rows, 1 column, with the first row 3 times the height of the second
# ax = fig.add_subplot(gs[0])
# # ax=axs[0]
# cax = ax.imshow(data, interpolation="nearest", cmap=cmap, vmin=1.015, vmax=1.045)
# # cax = sns.heatmap(data, cmap="viridis")

# # Add a colorbar
# fig.colorbar(cax, shrink=1)


# # Function to convert RGB to grayscale intensity
# def get_intensity(color):
#     return color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114


# # Set a threshold for deciding text color
# intensity_threshold = 0.5
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         # Get the color from the colormap
#         cell_color = cax.cmap(cax.norm(data[i, j]))
#         # Calculate intensity of the cell color
#         intensity = get_intensity(cell_color)
#         # Choose text color based on intensity
#         text_color = "white" if intensity < intensity_threshold else "black"
#         text = ax.text(
#             j, i, round(data[i, j], 3), ha="center", va="center", color=text_color
#         )

# # Set the x-axis and y-axis values
# x_axis_labels = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
# y_axis_labels = [256, 512, 1024, 2048]

# # Set ticks positions
# ax.set_xticks(np.arange(len(x_axis_labels)))
# ax.set_yticks(np.arange(len(y_axis_labels)))


# # Set ticks labels
# ax.set_xticklabels(x_axis_labels)
# ax.set_yticklabels(y_axis_labels)


# # Set labels for axes
# ax.set_xlabel("Output Length\n" + r"$\mathbf{Normalized\ TBT}$")
# ax.set_ylabel("Input Length")
# ax.invert_yaxis()
# # # Rotate the tick labels for the x-axis if needed
# # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


# # fig = plt.figure(figsize=(10, 5))  # Define the figure size
# axs1 = fig.add_subplot(gs[0, 1])
# axs1.barh(np.arange(len(data_ttft)) / 2 + 0.2, data_ttft, color="steelblue", height=0.3)
# axs1.set_yticks(np.arange(len(y_axis_labels)) / 2 + 0.2)
# axs1.set_yticklabels(y_axis_labels)
# axs1.set_xlabel(r"$\mathbf{Normalized\ TTFT}$")
# axs1.set_xlim(1, 2)

# # Show the plot
# plt.tight_layout()
# plt.savefig("figure11.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
