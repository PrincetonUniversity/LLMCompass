from utils import size, closest_factors
from typing import List, Tuple, Union
from hardware_model.device import Device
from software_model.utils import Tensor, DataType


class Operator:
    def __init__(
        self,
        flop_count,
        load_count,
        store_count,
        peak_memory_usage,
        data_type: DataType,
        gpu_device=None,
        verbose=True,
    ):
        self.flop_count = flop_count
        self.load_count = load_count
        self.store_count = store_count
        self.io_count = load_count + store_count
        self.peak_memory_usage = peak_memory_usage
        self.data_type = data_type
        self.gpu_device = gpu_device
        self.verbose = verbose
        self.log = ""
        self.comment = ""
        # simulation results
        self.latency = 0
        self.latency_on_gpu = 1
        self.is_io_bound = None
        # run on gpu
        self.iterations = 50

    class mapping:
        pass


# auxilary functions


class Reshape(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input_shape = None
        self.output_shape = None

    def __call__(self, input: Tensor, output_shape: List[int]) -> Tensor:
        assert input.size == size(output_shape)
        self.flop_count = 0
        self.load_count = 0
        self.store_count = 0
        self.io_count = 0
        self.peak_memory_usage = 0
        self.input_shape = input.shape
        self.output_shape = output_shape
        output = Tensor(output_shape, self.data_type)
        return output


class Concat(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.concat_dim = None
        self.output_shape = None

    def __call__(self, input1: Tensor, input2: Tensor, concat_dim: int) -> Tensor:
        assert len(input1.shape) == len(input2.shape)
        for i in range(len(input1.shape)):
            if i != concat_dim:
                assert input1.shape[i] == input2.shape[i]
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.concat_dim = concat_dim
        self.flop_count = 0
        self.load_count = input1.size + input2.size
        self.store_count = input1.size + input2.size
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = (input1.size + input2.size) * 2
        self.output_shape = (
            input1.shape[:concat_dim]
            + [input1.shape[concat_dim] + input2.shape[concat_dim]]
            + input1.shape[concat_dim + 1 :]
        )
        output = Tensor(self.output_shape, self.data_type)
        return output


class Transpose(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input_shape = None
        self.output_shape = None

    def __call__(self, input: Tensor, permute: List[int]) -> Tensor:
        assert len(input.shape) == len(permute)
        self.input_shape = input.shape
        self.permute = permute

        self.flop_count = 0
        self.load_count = size(input.shape)
        self.store_count = self.load_count
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = input.size * 2

        self.output_shape = [self.input_shape[i] for i in permute]
        output = Tensor(self.output_shape, self.data_type)
        return output
