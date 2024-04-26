from software_model.softmax import Softmax
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simamd", action="store_true", help="amd simulation")
    parser.add_argument("--roofline", action="store_true", help="Roofline simulation")
    args = parser.parse_args()

    A100 = device_dict["A100_80GB_fp16"]
    TPU = device_dict["TPUv3"]
    MI210 = device_dict["MI210"]
    tpu_overhead = 300e-6
    gpu_overhead = 12e-6
    amd_overhead = MI210.compute_module.overhead.softmax

    if args.gpu:
        gpu_kernel_launch_overhead = Softmax.gpu_kernel_launch_overhead()

    print(f"Performance of Softmax")
    M = 2**12
    for N in range(5, 16):
        N = 2**N
        if args.simtpu:
            model = Softmax(data_type=data_type_dict["fp32"])
            _ = model(Tensor([M, N], data_type=data_type_dict["fp32"]))
            if args.roofline:
                latency = model.roofline_model(pcb_module=TPU) + tpu_overhead
                file_name = "softmax_TPUv3_roofline.csv"
            else:
                latency = model.compile_and_simulate(pcb_module=TPU) + tpu_overhead
                file_name = "softmax_TPUv3_sim.csv"
        else:
            model = Softmax(data_type=data_type_dict["fp16"])
            _ = model(
                Tensor([M, N]),
            )
            if args.gpu:
                latency = model.run_on_gpu()
            if args.simgpu:
                if args.roofline:
                    latency = model.roofline_model(pcb_module=A100) + gpu_overhead
                    file_name = "softmax_A100_roofline.csv"
                else:
                    latency = model.compile_and_simulate(pcb_module=A100) + gpu_overhead
                    file_name = "softmax_A100_sim.csv"
            if args.simamd:
                model = Softmax(data_type=data_type_dict["fp32"])
                _ = model(Tensor([M, N], data_type=data_type_dict["fp32"]))
                if args.roofline:
                    latency = model.roofline_model(pcb_module=MI210) + amd_overhead
                    file_name = "softmax_MI210_roofline.csv"
                else:
                    latency = (
                        model.compile_and_simulate(pcb_module=MI210) + amd_overhead
                    )
                    file_name = "softmax_MI210_sim.csv"

        print(f"{M}, {N}, {M*N/latency/1e9}")
        with open(f"ae/figure5/cf/{file_name}", "a") as f:
            f.write(f"{M}, {N}, {M*N/latency/1e9}\n")

    N = 2**12
    for M in range(5, 16):
        M = 2**M
        if args.simtpu:
            model = Softmax(data_type=data_type_dict["fp32"])
            _ = model(Tensor([M, N], data_type=data_type_dict["fp32"]))
            if args.roofline:
                latency = model.roofline_model(pcb_module=TPU) + tpu_overhead
            else:
                latency = model.compile_and_simulate(pcb_module=TPU) + tpu_overhead
        else:
            model = Softmax(data_type=data_type_dict["fp16"])
            _ = model(
                Tensor([M, N]),
            )
            if args.gpu:
                latency = model.run_on_gpu()
            if args.simgpu:
                if args.roofline:
                    latency = model.roofline_model(pcb_module=A100) + gpu_overhead
                else:
                    latency = model.compile_and_simulate(pcb_module=A100) + gpu_overhead
            if args.simamd:
                model = Softmax(data_type=data_type_dict["fp32"])
                _ = model(Tensor([M, N], data_type=data_type_dict["fp32"]))
                if args.roofline:
                    latency = model.roofline_model(pcb_module=MI210) + amd_overhead
                else:
                    latency = (
                        model.compile_and_simulate(pcb_module=MI210) + amd_overhead
                    )
        print(f"{M}, {N}, {M*N/latency/1e9}")
        with open(f"ae/figure5/cf/{file_name}", "a") as f:
            f.write(f"{M}, {N}, {M*N/latency/1e9}\n")
