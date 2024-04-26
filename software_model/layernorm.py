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
def layernorm_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.layer_norm(input, [input.shape[-1]])


class LayerNorm(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
        return input

    def roofline_model(self, pcb_module: Device):
        self.io_count = self.M * self.N * self.data_type.word_size * 2
        self.flop_count = self.M * self.N * 7
        self.roofline_latency = max(
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
            self.flop_count / pcb_module.compute_module.total_vector_flops,
        )
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l1_tile_M: int,
            l1_tile_N: int,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}"
            )

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        min_cycle_count = float("inf")
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = (
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size) // 2
        )
        l2_tile_M = min(l2_tile_M, M)
        if compile_mode == "heuristic-GPU" or compile_mode == "heuristic-our-throughput":
            # if N <= 1024:
            l1_tile_N = N
            l1_tile_M = (
                pcb_module.compute_module.core.SRAM_size
                // (l1_tile_N * data_type.word_size)
                // 2
            )
            while l1_tile_M < pcb_module.compute_module.core.vector_unit.vector_count:
                l1_tile_N = l1_tile_N // 2
                l1_tile_M = (
                    pcb_module.compute_module.core.SRAM_size
                    // (l1_tile_N * data_type.word_size)
                    // 2
                )
            l1_tile_M = min(l1_tile_M, l2_tile_M)
        elif compile_mode == "heuristic-TPU":
            l1_tile_N = N
            l1_tile_M = pcb_module.compute_module.core.SRAM_size // (
                2 * l1_tile_N * data_type.word_size
            )
            l1_tile_M = min(l1_tile_M, M)
        mapping = self.Mapping(
            l2_tile_M,
            l2_tile_N,
            l1_tile_M,
            l1_tile_N,
        )
        cycle_count = self.simulate(self.computational_graph, mapping, pcb_module)
        if cycle_count < min_cycle_count:
            min_cycle_count = cycle_count
            best_mapping = mapping
        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        M = computational_graph.M
        N = computational_graph.N
        data_type = computational_graph.data_type
        l2_tile_M = mapping.l2_tile_M

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            )

        total_cycle_count = 0
        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(l2_tile_count):
            total_cycle_count += l2_tiles[m].read_cycle_count
            total_cycle_count += l2_tiles[m].compute_cycle_count
            total_cycle_count += l2_tiles[m].write_cycle_count
        return total_cycle_count

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.write_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = LayerNorm.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
            l1_tile_cycle_count = (
                l1_tile.read_cycle_count * 3
                + l1_tile.write_cycle_count
                + l1_tile.compute_cycle_count
            )
            total_cycle_count = (
                ceil(l1_tile_count / pcb_module.compute_module.core_count)
            ) * (
                l1_tile_cycle_count
                + (ceil(N / l1_tile_N) - 1) * (l1_tile.reduction_cycle_count)
            )
            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.write_cycle_count = self.simulate_l1_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.reduction_cycle_count = (
                M
                * N
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                + M
                * N
                * data_type.word_size
                * 2
                / (
                    pcb_module.compute_module.l2_bandwidth_per_cycle
                    / pcb_module.compute_module.core_count
                )
            )

        def simulate_l1_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, pcb_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (pcb_module.compute_module.l2_bandwidth_per_cycle)
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "LayerNorm.Mapping",
            pcb_module: Device,
        ):
            M_per_vector_count = ceil(
                M / pcb_module.compute_module.core.vector_unit.vector_count
            )
            N_per_vector_count = N
            M_per_vector_lane = M_per_vector_count
            N_per_vector_lane = ceil(
                N_per_vector_count
                / pcb_module.compute_module.core.vector_unit.vector_width
            )

            # each lane computes it own mean
            total_cycle_count = ceil(
                N_per_vector_lane
                * M_per_vector_lane
                / pcb_module.compute_module.core.vector_unit.flops_per_cycle
            )
            # the whole vector reduce to one mean
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            # each lane computes it own variance
            total_cycle_count += (
                ceil(
                    N_per_vector_lane
                    * M_per_vector_lane
                    / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                )
                * 2
            )
            # the whole vector reduce to one variance
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            # calculate normalized output
            total_cycle_count += (
                ceil(
                    N_per_vector_lane
                    * M_per_vector_lane
                    / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                )
                * 4
            )  # division is heavy

            return total_cycle_count

    def run_on_gpu(self):
        # import torch
        # from apex.normalization.fused_layer_norm import FusedLayerNorm
        # from apex.contrib.layer_norm import FastLayerNorm
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # warmup
        for _ in range(3):
            _ = layernorm_gpu(input)

            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = layernorm_gpu(input)
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
        a = torch.randn(1, 1, 1, device="cuda")
        for _ in range(50):
            start = time.time()
            c = layernorm_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead
