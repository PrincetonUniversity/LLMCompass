import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

def read_csv(filename: str):
    numbers = []
    # Open the CSV file and read the numbers
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Since each row contains only one number, we use row[0]
            numbers.append(float(row[0]))
    return numbers


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
colors_matmul = sns.color_palette("flare_r", 6)
colors_normalization = sns.color_palette("summer", 3)
colors_gelu = sns.color_palette("pink", 1)
colors_allreduce = sns.color_palette("Blues_r", 2)
colors = colors_matmul + colors_normalization + colors_gelu + colors_allreduce
# values_simgpu = read_csv("transformer_A100_sim.csv")
values_simgpu = pd.read_csv("transformer_A100_sim.csv", header=None, names=categories, index_col=None).iloc[0].tolist()
print(values_simgpu)
values_gpu = read_csv("real_hardware/transformer_A100.csv")
# values_gpu_roofline = read_csv("transformer_A100_roofline.csv")
values_gpu_roofline = pd.read_csv("transformer_A100_roofline.csv", header=None, names=categories, index_col=None).iloc[0].tolist()
# values_simtpu = read_csv("transformer_TPUv3_sim.csv")
values_simtpu = pd.read_csv("transformer_TPUv3_sim.csv", header=None, names=categories, index_col=None).iloc[0].tolist()

# values_tpu_roofline = read_csv("transformer_TPUv3_roofline.csv")
values_tpu_roofline = pd.read_csv("transformer_TPUv3_roofline.csv", header=None, names=categories, index_col=None).iloc[0].tolist()

plt.figure(figsize=(3, 2.8))

# Create the stacked bar graph
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_gpu)):
    plt.bar(1, value, bottom=bottom, color=colors[i], label=category, width=0.5)
    bottom += value
value_gt = bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_simgpu)):
    plt.bar(2, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_sim = bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_gpu_roofline)):
    plt.bar(3, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_roofline = bottom

print(f"gpu prefilling: {value_sim/value_gt}, {value_roofline/value_gt}")
# Set the title, legend, and display the graph
# plt.title(
#     "GPU Runtime Breakdown of One Transformer Layer in GPT-3 \n(Initial computation, batch size = 8, sequence length = 2048)"
# )
plt.ylabel("Latency (s)")
# plt.xlabel('Bar Sets')
plt.xticks([1, 2, 3], ["Real\nA100", "Simulated\nA100", "Roofline\nModel"])
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("figure5i.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


plt.figure(figsize=(3, 2.8))

# Create the stacked bar graph
# bottom = 0
# for i, (category, value) in enumerate(zip(categories, values_tpu)):
#     plt.bar(1, value, bottom=bottom, color=colors[i], label=category, width=0.5)
#     bottom += value
# value_gt = bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_simtpu)):
    plt.bar(2, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_sim = bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_tpu_roofline)):
    plt.bar(3, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_roofline = bottom
# print(f"tpu prefilling: {value_sim/value_gt}, {value_roofline/value_gt}")

# Set the title, legend, and display the graph
# plt.title(
#     "TPU Runtime Breakdown of One Transformer Layer in GPT-3 \n(Initial computation, batch size = 8, sequence length = 2048)"
# )
plt.ylabel("Latency (s)")
# plt.xlabel('Bar Sets')
plt.xticks([1, 2, 3], ["Real\nTPUv3", "Simulated\nTPUv3", "Roofline\nModel"])
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("figure5j.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


# values_simgpu = read_csv("transformerAR_A100_sim.csv")
values_simgpu=pd.read_csv("transformerAR_A100_sim.csv",header=None,names=categories,index_col=None).iloc[0].tolist()
values_gpu = read_csv("real_hardware/transformerAR_A100.csv")
# values_gpu_roofline = read_csv("transformerAR_A100_roofline.csv")
values_gpu_roofline=pd.read_csv("transformerAR_A100_roofline.csv",header=None,names=categories,index_col=None).iloc[0].tolist()

plt.figure(figsize=(3, 2.8))

# Create the stacked bar graph
bottom = 0
for i, (category, value) in enumerate(zip(categories, values_gpu)):
    value = value * 1e3
    plt.bar(1, value, bottom=bottom, color=colors[i], label=category, width=0.5)
    bottom += value
value_gt = bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_simgpu)):
    value = value * 1e3
    plt.bar(2, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_sim = bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_gpu_roofline)):
    value = value * 1e3
    plt.bar(3, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_roofline = bottom

print(value_sim / value_gt, value_roofline / value_gt)

# Set the title, legend, and display the graph
# plt.title(
#     "GPU Runtime Breakdown of One Transformer Layer in GPT-3 \n(Auto regression, batch size = 8, sequence length = 2048)"
# )
plt.ylabel("Latency (ms)")
# plt.xlabel('Bar Sets')
plt.xticks([1, 2, 3], ["Real\nA100", "Simulated\nA100", "Roofline\nModel"])
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("figure5k.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)




# values_simtpu = read_csv("transformerAR_TPUv3_sim.csv")
values_simtpu = pd.read_csv("transformerAR_TPUv3_sim.csv", header=None, names=categories, index_col=None).iloc[0].tolist()
# values_tpu = read_csv("real_hardware/transformerAR_TPUv3.csv")
# values_tpu_roofline=read_csv("transformerAR_TPUv3_roofline.csv")
values_tpu_roofline=pd.read_csv("transformerAR_TPUv3_roofline.csv",header=None,names=categories,index_col=None).iloc[0].tolist()

plt.figure(figsize=(4.5, 2.8))

# Create the stacked bar graph
# bottom = 0
# for i, (category, value) in enumerate(zip(categories, values_tpu)):
#     value=value*1e3
#     plt.bar(1, value, bottom=bottom, color=colors[i], label=category, width=0.5)
#     bottom += value
# value_gt=bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_simtpu)):
    value=value*1e3
    plt.bar(2, value, bottom=bottom, color=colors[i], label=category,width=0.5)
    bottom += value
value_sim=bottom

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_tpu_roofline)):
    value=value*1e3
    plt.bar(3, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
value_roofline=bottom
print(value_sim/value_gt,value_roofline/value_gt)

# Set the title, legend, and display the graph
# plt.title(
#     "GPU Runtime Breakdown of One Transformer Layer in GPT-3 \n(Auto regression, batch size = 8, input(output) sequence length = 2048(1024))"
# )
plt.ylabel("Latency (ms)")
# plt.xlabel('Bar Sets')
plt.xticks([1, 2, 3], ["Real\nTPUv3", "Simulated\nTPUv3", "Roofline\nModel"])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1.05))
plt.tight_layout()
plt.savefig("figure5l.pdf",bbox_inches="tight", pad_inches=0.01, dpi=300)