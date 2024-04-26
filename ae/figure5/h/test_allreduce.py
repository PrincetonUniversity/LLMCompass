from software_model.communication_primitives import AllReduceMultiPCB
from software_model.utils import data_type_dict, Tensor
from hardware_model.interconnect import interconnect_module_dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    interconnect_module = interconnect_module_dict["NVLinkV3_FC_4"]
    gpu_latency_list = [
        12.52,
        13.92,
        12.39,
        13.22,
        12.35,
        12.45,
        13.12,
        13.02,
        15.12,
        15.23,
        15.99,
        17.39,
        20.00,
        22.93,
        28.66,
        35.93,
        47.27,
        60.75,
        66.40,
        84.75,
        128.8,
        195.7,
        279.7,
        532.3,
        961.7,
        1883.7,
        3659.0,
        7219.2,
        14136,
        27944,
        55384,
        110277,
    ]
    size_list = [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        67108864,
        134217728,
        268435456,
        536870912,
        1073741824,
        2147483648,
        4294967296,
        8589934592,
        17179869184,
    ]
    simulated_latency_list = []
    data_type = data_type_dict["fp16"]
    for data_size in size_list:
        model = AllReduceMultiPCB(data_type=data_type)
        _ = model(
            Tensor([data_size / 2]),
        )
        our_latency = model.simulate(interconnect_module=interconnect_module)
        simulated_latency_list.append(our_latency * 1e6)

    gpu_bandwidth_list = np.array(size_list) / np.array(gpu_latency_list) / 1e3
    simulated_gpu_bandwidth_list = (
        np.array(size_list) / np.array(simulated_latency_list) / 1e3
    )

    size_list = size_list[9:]
    gpu_bandwidth_list = gpu_bandwidth_list[9:]
    simulated_gpu_bandwidth_list = simulated_gpu_bandwidth_list[9:]

    color_NV = sns.color_palette("Greens_d", 4)[1:]
    color_Google = sns.color_palette("Blues_d", 4)[1:]

    plt.figure(figsize=(6, 2.3))
    plt.xscale("log", base=2)
    plt.plot(
        size_list,
        gpu_bandwidth_list,
        marker="o",
        label="Real NVIDIA A100 Node",
        color=color_NV[0],
    )
    plt.plot(
        size_list,
        simulated_gpu_bandwidth_list,
        marker="x",
        label="Simulated NVIDIA A100 Node",
        color=color_NV[2],
    )

    interconnect_module = interconnect_module_dict["TPUv3Link_8"]
    simulated_tpu_bandwidth_list = []
    data_type = data_type_dict["fp16"]
    for data_size in size_list:
        model = AllReduceMultiPCB(data_type=data_type)
        _ = model(
            Tensor([data_size // 2]),
        )
        our_latency = model.simulate(interconnect_module=interconnect_module)
        simulated_tpu_bandwidth_list.append(data_size / our_latency / 1e9)

    
    plt.plot(
        size_list,
        simulated_tpu_bandwidth_list,
        marker="x",
        label="Simulated Google TPU v3 Node",
        color=color_Google[2],
    )
    plt.xlabel("Data Size (Bytes)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.grid(
        True, which="both", ls="--", c="0.7"
    )  # Adding a grid for better readability
    plt.legend()
    plt.savefig(
        "ae/figure5/h/figure5h.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300
    )
