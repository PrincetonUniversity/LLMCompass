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
arch_specs = read_architecture_template("configs/GA100.json")
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


def test_core_size(core_configs, lock):
    name, core_count, sublane_count, array_width, vector_width, sram_KB = core_configs
    arch_specs["device"]["compute_chiplet"]["core_count"] = core_count
    arch_specs["device"]["compute_chiplet"]["core"]["sublane_count"] = sublane_count
    arch_specs["device"]["compute_chiplet"]["core"]["systolic_array"][
        "array_width"
    ] = array_width
    arch_specs["device"]["compute_chiplet"]["core"]["systolic_array"][
        "array_height"
    ] = array_width
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"][
        "vector_width"
    ] = vector_width
    arch_specs["device"]["compute_chiplet"]["core"]["SRAM_KB"] = sram_KB
    # for area
    arch_specs["device"]["compute_chiplet"]["physical_core_count"] = core_count
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"]["int32_count"] = (
        vector_width // 2
    )
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"]["fp32_count"] = (
        vector_width // 2
    )
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"]["fp64_count"] = (
        vector_width // 4
    )
    if vector_width <= 32:
        arch_specs["device"]["compute_chiplet"]["core"]["register_file"][
            "num_registers"
        ] = (vector_width * 512)
    else:
        arch_specs["device"]["compute_chiplet"]["core"]["register_file"][
            "num_reg_files"
        ] = (vector_width // 32)
    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)
    print(f"{name}, {compute_area_mm2}, {io_area_mm2}, {compute_area_mm2+io_area_mm2}")
    # exit()
    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")
    print(f"{name}, {init_latency_simulated}, {auto_regression_latency_simulated}")
    with lock:
        with open(f"ae/figure7/core_size_results_init.csv", "a") as f:
            f.write(
                f"{name}, {compute_area_mm2+io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(f"ae/figure7/core_size_results_ar.csv", "a") as f:
            f.write(
                f"{name}, {compute_area_mm2+io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


lock = Lock()
configs = [
    ("A", 128, 4, 8, 8, 192),
    ("B", 128, 4, 16, 32, 192),
    ("C", 128, 1, 32, 128, 192),
    ("D", 32, 1, 64, 512, 768),
    ("E", 8, 1, 128, 2048, 3072),
]

processes = [Process(target=test_core_size, args=(i, lock)) for i in configs]

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
