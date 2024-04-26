import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


softmax_TPUv3_sim = pd.read_csv(
    "softmax_TPUv3_sim.csv", header=None, names=["M", "N", "throughput"]
)
softmax_TPUv3_sim.set_index(["M", "N"], inplace=True)
softmax_TPUv3_roofline = pd.read_csv(
    "softmax_TPUv3_roofline.csv", header=None, names=["M", "N", "throughput"]
)
softmax_TPUv3_roofline.set_index(["M", "N"], inplace=True)
softmax_A100 = pd.read_csv(
    "real_hardware/softmax_A100.csv", header=None, names=["M", "N", "throughput"]
)
softmax_A100.set_index(["M", "N"], inplace=True)
softmax_A100_sim = pd.read_csv(
    "softmax_A100_sim.csv", header=None, names=["M", "N", "throughput"]
)
softmax_A100_sim.set_index(["M", "N"], inplace=True)
softmax_A100_roofline = pd.read_csv(
    "softmax_A100_roofline.csv", header=None, names=["M", "N", "throughput"]
)
softmax_A100_roofline.set_index(["M", "N"], inplace=True)
softmax_MI210 = pd.read_csv(
    "real_hardware/softmax_MI210.csv", header=None, names=["M", "N", "throughput"]
)
softmax_MI210.set_index(["M", "N"], inplace=True)
softmax_MI210_sim = pd.read_csv(
    "softmax_MI210_sim.csv", header=None, names=["M", "N", "throughput"]
)
softmax_MI210_sim.set_index(["M", "N"], inplace=True)
softmax_MI210_roofline = pd.read_csv(
    "softmax_MI210_roofline.csv", header=None, names=["M", "N", "throughput"]
)
softmax_MI210_roofline.set_index(["M", "N"], inplace=True)

color_NV = sns.color_palette("Greens_d", 4)[1:]
color_Google = sns.color_palette("Blues_d", 4)[1:]
color_AMD = sns.color_palette("flare", 3)

M = 4096
title = f"Performance of softmax with M={M}"
N_list = []
throughput_TPU_list = []
throughput_TPU_sim_list = []
throughput_TPU_roofline_list = []
throughput_GPU_list = []
throughput_GPU_sim_list = []
throughput_GPU_roofline_list = []
throughput_AMD_list = []
throughput_AMD_sim_list = []
throughput_AMD_roofline_list = []
for N in range(6, 16):
    N = 2**N
    N_list.append(N)
    # print(M,N)
    # print(softmax_TPUv3.loc[(M, N), 'throughput'])

    throughput_TPU_sim_list.append(
        softmax_TPUv3_sim.loc[(M, N), "throughput"].values[0]
    )
    throughput_TPU_roofline_list.append(
        softmax_TPUv3_roofline.loc[(M, N), "throughput"].values[0]
    )
    throughput_GPU_list.append(softmax_A100.loc[(M, N), "throughput"].values[0])
    throughput_GPU_sim_list.append(softmax_A100_sim.loc[(M, N), "throughput"].values[0])
    throughput_GPU_roofline_list.append(
        softmax_A100_roofline.loc[(M, N), "throughput"].values[0]
    )
    throughput_AMD_list.append(softmax_MI210.loc[(M, N), "throughput"].values[0])
    throughput_AMD_sim_list.append(
        softmax_MI210_sim.loc[(M, N), "throughput"].values[0]
    )
    throughput_AMD_roofline_list.append(
        softmax_MI210_roofline.loc[(M, N), "throughput"].values[0]
    )

plt.figure(figsize=(3.7, 2))
plt.xscale("log", base=2)
plt.plot(
    N_list,
    throughput_GPU_roofline_list,
    marker=" ",
    linewidth=1.5,
    linestyle="--",
    label="Roofline of NVIDIA A100",
    color=color_NV[0],
)
plt.plot(
    N_list, throughput_GPU_list, marker="o", label="Real NVIDIA A100", color=color_NV[1]
)
plt.plot(
    N_list,
    throughput_GPU_sim_list,
    marker="x",
    label="Simulated NVIDIA A100",
    color=color_NV[2],
)
plt.plot(
    N_list,
    throughput_AMD_roofline_list,
    marker=" ",
    linewidth=1.5,
    linestyle="--",
    label="Roofline of AMD MI210",
    color=color_AMD[0],
)
plt.plot(
    N_list, throughput_AMD_list, marker="o", label="Real AMD MI210", color=color_AMD[1]
)
plt.plot(
    N_list,
    throughput_AMD_sim_list,
    marker="x",
    label="Simulated AMD MI210",
    color=color_AMD[2],
)
plt.plot(
    N_list,
    throughput_TPU_roofline_list,
    marker=" ",
    linewidth=1.5,
    linestyle="--",
    label="Roofline of Google TPUv3",
    color=color_Google[0],
)

plt.plot(
    N_list,
    throughput_TPU_sim_list,
    marker="x",
    label="Simulated Google TPUv3",
    color=color_Google[2],
)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1.1))
# plt.legend()
# plt.title(title)
plt.xlabel("N")
plt.ylabel("G Elements/s")
plt.grid(True, which="both", ls="--", c="0.7")  # Adding a grid for better readability
plt.savefig("figure5f.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


N = 4096
title = f"Performance of softmax with N={N}"
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
for M in range(6, 16):
    M = 2**M
    M_list.append(M)
    throughput_TPU_sim_list.append(
        softmax_TPUv3_sim.loc[(M, N), "throughput"].values[0]
    )
    throughput_TPU_roofline_list.append(
        softmax_TPUv3_roofline.loc[(M, N), "throughput"].values[0]
    )
    throughput_GPU_list.append(softmax_A100.loc[(M, N), "throughput"].values[0])
    throughput_GPU_sim_list.append(softmax_A100_sim.loc[(M, N), "throughput"].values[0])
    throughput_GPU_roofline_list.append(
        softmax_A100_roofline.loc[(M, N), "throughput"].values[0]
    )
    throughput_AMD_list.append(softmax_MI210.loc[(M, N), "throughput"].values[0])
    throughput_AMD_sim_list.append(
        softmax_MI210_sim.loc[(M, N), "throughput"].values[0]
    )
    throughput_AMD_roofline_list.append(
        softmax_MI210_roofline.loc[(M, N), "throughput"].values[0]
    )


plt.figure(figsize=(3.7, 2.8))
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
    linewidth=1.5,
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
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
# plt.legend()
# plt.title(title)
plt.xlabel("M")
plt.ylabel("G Elements/s")
plt.grid(True, which="both", ls="--", c="0.7")  # Adding a grid for better readability
plt.savefig("figure5c.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
