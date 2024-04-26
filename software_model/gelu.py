from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, log
import time
import statistics
import numpy as np
import torch


@torch.compile
def gelu_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(input, approximate="tanh")


# x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
class GeLU(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:])
        self.computational_graph = self.ComputationalGraph(self.M, self.data_type)
        return input

    def roofline_model(self, pcb_module: Device):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        M = self.M
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / min(pcb_module.io_module.bandwidth
            , pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq)
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, data_type: DataType):            
            self.M = M
            self.data_type = data_type

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        parallelism = (
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.vector_unit.vector_width
            * pcb_module.compute_module.core.vector_unit.vector_count
        )
        M = ceil(self.computational_graph.M / parallelism) * parallelism
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )

        return max(compute_latency, io_latency)

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # warmup
        for _ in range(3):
            _ = gelu_gpu(input)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = gelu_gpu(input)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
            latencies.append(end - start)
        # print(latencies)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        import torch

        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = gelu_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead
