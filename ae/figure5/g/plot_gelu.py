import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


gelu_TPUv3_sim = pd.read_csv(
    "gelu_TPUv3_sim.csv", header=None, names=["M", "throughput"]
)
gelu_TPUv3_roofline = pd.read_csv(
    "gelu_TPUv3_roofline.csv", header=None, names=["M", "throughput"]
)
gelu_A100 = pd.read_csv("real_hardware/gelu_A100.csv", header=None, names=["M", "throughput"])
gelu_A100_sim = pd.read_csv("gelu_A100_sim.csv", header=None, names=["M", "throughput"])
gelu_A100_roofline = pd.read_csv(
    "gelu_A100_roofline.csv", header=None, names=["M", "throughput"]
)
gelu_MI210 = pd.read_csv("real_hardware/gelu_MI210.csv", header=None, names=["M", "throughput"])
gelu_MI210_sim = pd.read_csv(
    "gelu_MI210_sim.csv", header=None, names=["M", "throughput"]
)
gelu_MI210_roofline = pd.read_csv(
    "gelu_MI210_roofline.csv", header=None, names=["M", "throughput"]
)

color_NV = sns.color_palette("Greens_d", 4)[1:]
color_Google = sns.color_palette("Blues_d", 4)[1:]
color_AMD = sns.color_palette("flare", 3)

M = 4096
title = f"Performance of gelu with M={M}"
M_list = []
throughput_TPU_list = []
throughput_TPU_sim_list = []
throughput_TPU_roofline_list = []
throughput_GPU_list = []
throughput_GPU_sim_list = []
throughput_GPU_roofline_list = []
throughput_AMD_list = []
throughput_AMD_sim_list = []
throughput_AMD_roofline_list = []
for M in range(10, 30):
    M = 2**M
    M_list.append(M)
    
    throughput_TPU_sim_list.append(
        gelu_TPUv3_sim[gelu_TPUv3_sim["M"] == M]["throughput"].iloc[0]
    )
    throughput_TPU_roofline_list.append(
        gelu_TPUv3_roofline[gelu_TPUv3_roofline["M"] == M]["throughput"].iloc[0]
    )
    throughput_GPU_list.append(gelu_A100[gelu_A100["M"] == M]["throughput"].iloc[0])
    throughput_GPU_sim_list.append(
        gelu_A100_sim[gelu_A100_sim["M"] == M]["throughput"].iloc[0]
    )
    throughput_GPU_roofline_list.append(
        gelu_A100_roofline[gelu_A100_roofline["M"] == M]["throughput"].iloc[0]
    )
    throughput_AMD_list.append(gelu_MI210[gelu_MI210["M"] == M]["throughput"].iloc[0])
    throughput_AMD_sim_list.append(
        gelu_MI210_sim[gelu_MI210_sim["M"] == M]["throughput"].iloc[0]
    )
    throughput_AMD_roofline_list.append(
        gelu_MI210_roofline[gelu_MI210_roofline["M"] == M]["throughput"].iloc[0]
    )

plt.figure(figsize=(6, 2.3))
plt.xscale("log", base=2)
plt.plot(
    M_list,
    throughput_GPU_roofline_list,
    marker=" ",
    linewidth=1.5,
    linestyle="--",
    label="Roofline of NVIDIA A100",
    color=color_NV[0],
)
plt.plot(
    M_list, throughput_GPU_list, marker="o", label="Real NVIDIA A100", color=color_NV[1]
)
plt.plot(
    M_list,
    throughput_GPU_sim_list,
    marker="x",
    label="Simulated NVIDIA A100",
    color=color_NV[2],
)
plt.plot(
    M_list,
    throughput_AMD_roofline_list,
    marker=" ",
    linewidth=1.5,
    linestyle="--",
    label="Roofline of AMD MI210",
    color=color_AMD[0],
)
plt.plot(
    M_list, throughput_AMD_list, marker="o", label="Real AMD MI210", color=color_AMD[1]
)
plt.plot(
    M_list,
    throughput_AMD_sim_list,
    marker="x",
    label="Simulated AMD MI210",
    color=color_AMD[2],
)
plt.plot(
    M_list,
    throughput_TPU_roofline_list,
    marker=" ",
    linewidth=3.5,
    linestyle="--",
    label="Roofline of Google TPUv3",
    color=color_Google[0],
)
plt.plot(
    M_list,
    throughput_TPU_sim_list,
    marker="x",
    label="Simulated Google TPUv3",
    color=color_Google[2],
)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1.05))
# plt.title(title)
plt.xlabel("# Elements")
plt.ylabel("G Elements/s")
plt.grid(True, which="both", ls="--", c="0.7")  # Adding a grid for better readability
plt.savefig(f"figure5g.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
