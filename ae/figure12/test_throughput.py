from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from design_space_exploration.dse import template_to_system, read_architecture_template
from multiprocessing import Process, Lock
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
import time

A100_specs = read_architecture_template("configs/GA100.json")
A100_system = template_to_system(A100_specs)
our_specs = read_architecture_template("configs/prefilling_system.json")
our_system = template_to_system(our_specs)
A100_compute_area_mm2 = calc_compute_chiplet_area_mm2(A100_specs)
A100_io_area_mm2 = calc_io_die_area_mm2(A100_specs)
our_compute_area_mm2 = calc_compute_chiplet_area_mm2(our_specs)
our_io_area_mm2 = calc_io_die_area_mm2(our_specs)
print(f"A100 compute area: {A100_compute_area_mm2} mm2")
print(f"A100 IO area: {A100_io_area_mm2} mm2")
print(f"A100 total area: {A100_compute_area_mm2+A100_io_area_mm2} mm2")
print(f"Our compute area: {our_compute_area_mm2} mm2")
print(f"Our IO area: {our_io_area_mm2} mm2")
print(f"Our total area: {our_compute_area_mm2+our_io_area_mm2} mm2")
with open("ae/figure12/area.csv", "w") as f:
    f.write(f"A100 compute area: {A100_compute_area_mm2} mm2\n")
    f.write(f"A100 IO area: {A100_io_area_mm2} mm2\n")
    f.write(f"A100 total area: {A100_compute_area_mm2+A100_io_area_mm2} mm2\n")
    f.write(f"Our compute area: {our_compute_area_mm2} mm2\n")
    f.write(f"Our IO area: {our_io_area_mm2} mm2\n")
    f.write(f"Our total area: {our_compute_area_mm2+our_io_area_mm2} mm2\n")


def simulate_decoding_latency(system, bs, seq_len, name, lock, heuristics):
    model_auto_regression = TransformerBlockAutoRegressionTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    _ = model_auto_regression(
        Tensor([bs, 1, 12288], data_type_dict["fp16"]),
        seq_len,
    )
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, heuristics
    )
    with lock:
        with open(f"ae/figure12/{name}_decoding.csv", "a") as f:
            f.write(
                f"{bs}, {seq_len}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


def simulate_prefill_latency(system, bs, seq_len, name, lock, heuristics):
    model = TransformerBlockInitComputationTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    _ = model(
        Tensor([bs, seq_len, 12288], data_type_dict["fp16"]),
    )
    latency_simulated = model.compile_and_simulate(system, heuristics)
    with lock:
        with open(f"ae/figure12/{name}_prefill.csv", "a") as f:
            f.write(f"{bs}, {seq_len}, {latency_simulated}, {model.simluate_log}\n")


lock_our_prefill = Lock()
lock_our_decoding = Lock()
lock_A100_prefill = Lock()
lock_A100_decoding = Lock()

processes = []

for input_seq_len in [
    256,
    512,
    1024,
    2048,
]:
    for output_seq_len in [
        256,
        512,
        768,
        1024,
        1280,
        1536,
        1792,
        2048,
    ]:
        seq_len = input_seq_len + output_seq_len
        for system in [our_system, A100_system]:
            if system == A100_system:
                name = f"A100/A100_{input_seq_len}_{output_seq_len}"
                lock = lock_A100_prefill
                bs = (80e9 - 2 * 12 * 12288**2 * 12) // ((12 * 4 + 8) * seq_len * 12288)
                heuristics = "heuristic-GPU"
            else:
                name = f"our/our_{input_seq_len}_{output_seq_len}"
                lock = lock_our_prefill
                bs = (512e9 - 2 * 12 * 12288**2 * 12) // (
                    (12 * 4 + 8) * seq_len * 12288
                )
                heuristics = "heuristic-our-throughput"
            bs = int(bs)
            # print(bs)
            p = Process(
                target=simulate_prefill_latency,
                args=(system, bs, input_seq_len, name, lock, heuristics),
            )
            processes.append(p)

            for decoding_seq_len in range(input_seq_len, seq_len + 64, 64):
                if system == A100_system:
                    name = f"A100/A100_{input_seq_len}_{output_seq_len}"
                    lock = lock_A100_decoding
                    heuristics = "heuristic-GPU"
                else:
                    name = f"our/our_{input_seq_len}_{output_seq_len}"
                    lock = lock_our_decoding
                    heuristics = "heuristic-our-throughput"
                p = Process(
                    target=simulate_decoding_latency,
                    args=(system, bs, decoding_seq_len, name, lock, heuristics),
                )
                processes.append(p)

print(len(processes))
# exit()
try:
    for p in processes:
        p.start()
    print("Processes started.")
    print("number of process:", len(processes))
    while any(p.is_alive() for p in processes):
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating processes...")
    for p in processes:
        p.terminate()
        p.join()


print("All processes have finished.")
