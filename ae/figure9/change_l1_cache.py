from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from design_space_exploration.dse import template_to_system, read_architecture_template
from multiprocessing import Process, Lock
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
import time

input_seq_length = 2048
batch_size = 8
output_seq_length = 1024
arch_specs = read_architecture_template("configs/template.json")
device_count = arch_specs["device_count"]

model_init = TransformerBlockInitComputationTP(
    d_model=12288,
    n_heads=96,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)
model_auto_regression = TransformerBlockAutoRegressionTP(
    d_model=12288,
    n_heads=96,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)
_ = model_init(
    Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"])
)
_ = model_auto_regression(
    Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]),
    input_seq_length + output_seq_length,
)


def test_SRAM_KB(SRAM_KB, lock):
    arch_specs["device"]["compute_chiplet"]["core"]["SRAM_KB"] = SRAM_KB
    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)
    print(
        f"{SRAM_KB}, {compute_area_mm2}, {io_area_mm2}, {compute_area_mm2+io_area_mm2}"
    )
    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")
    print(f"{SRAM_KB}, {init_latency_simulated}, {auto_regression_latency_simulated}")
    with lock:
        with open(f"ae/figure9/l1_cache_results_init.csv", "a") as f:
            f.write(
                f"{SRAM_KB}, {compute_area_mm2+io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(f"ae/figure9/l1_cache_results_ar.csv", "a") as f:
            f.write(
                f"{SRAM_KB}, {compute_area_mm2+io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


# for SRAM_KB in [64, 128, 192, 256, 512, 1024]:
#     test_SRAM_KB(SRAM_KB, None)

lock = Lock()
processes = [
    Process(target=test_SRAM_KB, args=(i, lock)) for i in [64, 128, 192, 256, 512, 1024]
]

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

# for SRAM_KB in [64, 128, 192, 256, 512, 1024]:
#     arch_specs["device"]["compute_chiplet"]["core"][
#                                     "SRAM_KB"
#                                 ] = SRAM_KB
#     system=template_to_system(arch_specs)
#     auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(system, 'heuristic-GPU')
#     init_latency_simulated = model_init.compile_and_simulate(system, 'heuristic-GPU')
#     print(f'{SRAM_KB}, {init_latency_simulated}, {auto_regression_latency_simulated}')
#     with open(f'test/case_study/l1_cache/l1_cache_results.csv', 'a') as f:
#         f.write(f'{SRAM_KB}, {init_latency_simulated}, {auto_regression_latency_simulated}\n')
