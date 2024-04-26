from math import ceil
from software_model.utils import DataType, data_type_dict


class VectorUnit:
    def __init__(
        self,
        total_vector_flops_per_cycle,
        word_size,
        flops_per_exp,
        vector_width,
        vector_count,
        data_type=data_type_dict["fp16"],
    ):
        self.total_vector_flops_per_cycle = total_vector_flops_per_cycle
        self.word_size = word_size  # Byte
        self.flops_per_exp = flops_per_exp  # flops per exp instruction
        self.vector_width = vector_width
        self.vector_count = vector_count
        self.flops_per_cycle = ceil(
            total_vector_flops_per_cycle / vector_width / vector_count
        )
        self.data_type = data_type


vector_unit_dict = {
    "A100_fp16": VectorUnit(512, 2, 35, 32, 4),
    "TPUv3_fp32": VectorUnit(128 * 8, 4, 15, 128, 8, data_type_dict["fp32"]),
    "MI210_fp32": VectorUnit(128, 4, 18, 16, 4, data_type_dict["fp32"]),
    "TPUv3_new": VectorUnit(128 * 4, 4, 15, 128, 4, data_type_dict["fp32"]),
}


class SystolicArray:
    def __init__(
        self,
        array_height,
        array_width,
        mac_per_cycle,
        input_word_size,
        output_word_size,
    ):
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle = mac_per_cycle
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size


systolic_array_dict = {
    "A100_fp16": SystolicArray(16, 16, 1, 2, 2),
    "A100_int8": SystolicArray(16, 16, 2, 1, 4),
    "TPUv3_bf16": SystolicArray(128, 128, 1, 2, 4),
    "MI210_fp16": SystolicArray(16, 16, 0.5, 2, 2),
    "TPUv3_new": SystolicArray(128, 128, 1, 2, 4),
}


class Core:
    def __init__(
        self,
        vector_unit: VectorUnit,
        systolic_array: SystolicArray,
        systolic_array_count,
        SRAM_size,
    ):
        self.vector_unit = vector_unit
        self.systolic_array = systolic_array
        self.systolic_array_count = systolic_array_count
        self.SRAM_size = SRAM_size  # Byte
        # assert(vector_unit.word_size==systolic_array.word_size)
        self.vector_word_size = vector_unit.word_size


core_dict = {
    "SM_A100_fp16": Core(
        vector_unit_dict["A100_fp16"], systolic_array_dict["A100_fp16"], 4, 192 * 1024
    ),
    "SM_A100_int8": Core(
        vector_unit_dict["A100_fp16"], systolic_array_dict["A100_int8"], 4, 192 * 1024
    ),
    "Core_TPUv3_bf16": Core(
        vector_unit_dict["TPUv3_fp32"],
        systolic_array_dict["TPUv3_bf16"],
        2,
        16 * 1024 * 1024,
    ),
    "CU_MI210_fp16": Core(
        vector_unit_dict["MI210_fp32"], systolic_array_dict["MI210_fp16"], 4, 128 * 1024
    ),
    "Core_TPUv3_new": Core(
        vector_unit_dict["TPUv3_new"],
        systolic_array_dict["TPUv3_new"],
        1,
        8 * 1024 * 1024,
    ),
}
# compute_tile_dict={'SM_A100_int8':ComputeTile(512, 4096, 192*1024*8,3.41, 'TSMC N7', 128*8),'SM_A100_fp16':ComputeTile(512, 2048, 192*1024*8,3.41, 'TSMC N7', 128),}
# flops: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch__fig2
# area: https://pbs.twimg.com/media/FOT_-NJWUAARrtB?format=jpg&name=large


class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu


overhead_dict = {
    "A100": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "TPUv3": Overhead(11e-5, 30e-5, 14e-5, 10e-5),
    "MI210": Overhead(3.4e-5, 2.2e-5, 2.8e-5, 2.1e-5),
}


class ComputeModule:
    def __init__(
        self,
        core: Core,
        core_count,
        clock_freq,
        l2_size,
        l2_bandwidth_per_cycle,
        overhead: Overhead = overhead_dict["A100"],
    ):
        self.core = core
        self.core_count = core_count
        self.clock_freq = clock_freq
        self.l2_size = int(l2_size)  # Byte
        self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle  # Byte/clock
        self.total_vector_flops_per_cycle = (
            core.vector_unit.total_vector_flops_per_cycle * core_count
        )
        self.total_vector_flops = self.total_vector_flops_per_cycle * clock_freq
        self.total_systolic_array_flops = (
            core_count
            * core.systolic_array_count
            * core.systolic_array.mac_per_cycle
            * 2
            * core.systolic_array.array_height
            * core.systolic_array.array_width
            * clock_freq
        )
        self.overhead = overhead


compute_module_dict = {
    "A100_fp16": ComputeModule(
        core_dict["SM_A100_fp16"],
        108,
        1.41e9,
        40 * 1024**2,
        5120,
        overhead_dict["A100"],
    ),
    "A100_int8": ComputeModule(
        core_dict["SM_A100_int8"],
        108,
        1.41e9,
        40 * 1024**2,
        5120,
        overhead_dict["A100"],
    ),
    "TPUv3_bf16": ComputeModule(
        core_dict["Core_TPUv3_bf16"],
        1,
        940e6,
        16 * 1024**3,
        490,
        overhead_dict["TPUv3"],
    ),
    "MI210_fp16": ComputeModule(
        core_dict["CU_MI210_fp16"],
        104,
        1.4e9,
        8 * 1024**2,
        4096,
        overhead_dict["MI210"],
    ),
    "TPUv3_new": ComputeModule(
        core_dict["Core_TPUv3_new"],
        2,
        940e6,
        16 * 1024**3,
        490,
        overhead_dict["TPUv3"],
    ),
}
