from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simtpu-new", action="store_true", help="Enable simulation")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simamd", action="store_true", help="amd simulation")
    parser.add_argument("--roofline", action="store_true", help="Roofline simulation")
    args = parser.parse_args()

    if args.simtpu:
        pcb = device_dict["TPUv3"]
    if args.simtpu_new:
        pcb = device_dict["TPUv3_new"]
    if args.simgpu:
        pcb = device_dict["A100_80GB_fp16"]

    MI210 = device_dict["MI210"]
    amd_overhead = MI210.compute_module.overhead.softmax

    K = 12288
    N = K
    titile = f"Performance of Matmul with K={K}, N={N}"
    print(f"{titile}")

    test_overhead = True

    for M in range(5, 16):
        M = 2**M
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        if args.gpu:
            if test_overhead:
                model.gpu_kernel_launch_overhead()
                test_overhead = False
            latency = model.run_on_gpu()
        if args.simtpu:
            if args.roofline:
                latency = model.roofline_model(pcb) + 110e-6
                file_name='matmul_TPUv3_roofline.csv'
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-TPU")
                    + 110e-6
                )
                file_name='matmul_TPUv3_sim.csv'

        if args.simtpu_new:
            if args.roofline:
                latency = model.roofline_model(pcb) + 110e-6
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-TPU-new")
                    + 110e-6
                )
        if args.simgpu:
            if args.roofline:
                latency = model.roofline_model(pcb) + 2.1e-5
                file_name='matmul_A100_roofline.csv'
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-GPU")
                    + 2.1e-5
                )
                file_name='matmul_A100_sim.csv'
        if args.simamd:
            if args.roofline:
                latency = model.roofline_model(pcb_module=MI210) + amd_overhead
                file_name='matmul_MI210_roofline.csv'
            else:
                latency = (
                    model.compile_and_simulate(
                        pcb_module=MI210, compile_mode="heuristic-GPU"
                    )
                    + amd_overhead
                )
                file_name='matmul_MI210_sim.csv'
        tflops = 2 * M * N * K / latency / 1e12
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops", flush=True)
        with open(f'ae/figure5/ab/{file_name}', 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops\n")

    M = 8192
    print(f"Performance of Matmul with M={M}, N=K")
    for K in range(5, 16):
        K = 2**K
        N = K
        model = Matmul(data_type=data_type_dict["fp16"])
        _ = model(
            Tensor([M, K]),
            Tensor([K, N]),
        )
        if args.gpu:
            latency = model.run_on_gpu()
        if args.simtpu:
            if args.roofline:
                latency = model.roofline_model(pcb) + 110e-6
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-TPU")
                    + 110e-6
                )
        if args.simtpu_new:
            if args.roofline:
                latency = model.roofline_model(pcb) + 110e-6
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-TPU-new")
                    + 110e-6
                )
        if args.simgpu:
            if args.roofline:
                latency = model.roofline_model(pcb) + 2.1e-5
            else:
                latency = (
                    model.compile_and_simulate(pcb, compile_mode="heuristic-GPU")
                    + 2.1e-5
                )
        if args.simamd:
            if args.roofline:
                latency = model.roofline_model(pcb_module=MI210) + amd_overhead
            else:
                latency = (
                    model.compile_and_simulate(
                        pcb_module=MI210, compile_mode="heuristic-GPU"
                    )
                    + amd_overhead
                )
        tflops = 2 * M * N * K / latency / 1e12
        print(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops", flush=True)
        with open(f'ae/figure5/ab/{file_name}', 'a') as f:
            f.write(f"{M}, {N}, {K}, {latency*1e3:.4f}ms, {tflops:.4f}Tflops\n")
