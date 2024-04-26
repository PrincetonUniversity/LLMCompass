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
our_specs = read_architecture_template("configs/latency_design.json")
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


def simulate_latency(system, bs, seq_len, name, lock):
    model_auto_regression = TransformerBlockAutoRegressionTP(
        d_model=12288,
        n_heads=96,
        device_count=4,
        data_type=data_type_dict["fp16"],
    )
    _ = model_auto_regression(
        Tensor([bs, 1, 12288], data_type_dict["fp16"]),
        seq_len,
    )
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    with lock:
        with open(f"ae/figure11/{name}.csv", "a") as f:
            f.write(
                f"{bs}, {seq_len}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


lock = Lock()

processes = []
for bs in [1, 2, 4, 8, 16, 32, 64]:
    for seq_len in [512, 2048]:
        for system in [our_system, A100_system]:
            if system == A100_system:
                name = "A100"
            else:
                name = "our"
            p = Process(target=simulate_latency, args=(system, bs, seq_len, name, lock))
            processes.append(p)

try:
    for p in processes:
        p.start()

    while any(p.is_alive() for p in processes):
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating processes...")
    for p in processes:
        p.terminate()
        p.join()


print("All processes have finished.")
