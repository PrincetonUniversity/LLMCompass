from software_model.gelu import GeLU
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--roofline", action="store_true", help="Roofline simulation")
    parser.add_argument("--amd", action="store_true", help="Enable AMD")
    parser.add_argument("--simamd", action="store_true", help="Enable simulation")
    args = parser.parse_args()

    A100 = device_dict["A100_80GB_fp16"]
    TPU = device_dict["TPUv3"]
    MI210 = device_dict["MI210"]

    if args.gpu:
        gpu_kernel_launch_overhead = GeLU.gpu_kernel_launch_overhead()

    print(f"Performance of GELU")
    for M in range(10, 30):
        M = 2**M
        # N = 2**15
        if args.simtpu:
            model = GeLU(data_type=data_type_dict["fp32"])
            _ = model(Tensor([M], data_type=data_type_dict["fp32"]))
            if args.roofline:
                latency = model.roofline_model(TPU) + 100e-6
                file_name = "gelu_TPUv3_roofline.csv"
            else:
                latency = (
                    model.compile_and_simulate(
                        pcb_module=TPU, compile_mode="heuristic-TPU"
                    )
                    + 100e-6
                )
                file_name = "gelu_TPUv3_sim.csv"
        else:
            model = GeLU(data_type=data_type_dict["fp16"])
            _ = model(
                Tensor([M]),
            )
            if args.gpu:
                latency = model.run_on_gpu()
            if args.amd:
                model.amd_kernel_launch_overhead()
                latency = model.run_on_amd()
            if args.simgpu:
                if args.roofline:
                    latency = model.roofline_model(A100) + 4.5e-5
                    file_name = "gelu_A100_roofline.csv"
                else:
                    latency = (
                        model.compile_and_simulate(
                            pcb_module=A100, compile_mode="heuristic-GPU"
                        )
                        + 4.5e-5
                    )
                    file_name = "gelu_A100_sim.csv"
            if args.simamd:
                if args.roofline:
                    latency = (
                        model.roofline_model(MI210) + MI210.compute_module.overhead.gelu
                    )
                    file_name = "gelu_MI210_roofline.csv"
                else:
                    latency = (
                        model.compile_and_simulate(
                            pcb_module=MI210, compile_mode="heuristic-GPU"
                        )
                        + MI210.compute_module.overhead.gelu
                    )
                    file_name = "gelu_MI210_sim.csv"
        print(f"{M}, {M/latency/1e9}")
        with open(f"ae/figure5/g/{file_name}", "a") as f:
            f.write(f"{M}, {M/latency/1e9}\n")
