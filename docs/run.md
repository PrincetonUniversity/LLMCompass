# User Guide: How to Run a LLMCompass Simulation

## Step 1: Build a Hardware Configuration

Follow the [NVIDIA GA100 example](../configs/GA100.json). This is a 4-GA100 node connected with NVLinks.

### Explanations on the Knobs
Most of the attributes are self-explained:

```json
{
    "name": "NVIDIA A100(80GB)x4", 
    "device_count": 4, # how many devices in a node
    "interconnect": {
        "link": {
            "name": "NVLink3",
            "bandwidth_per_direction_byte": 25e9,
            "bandwidth_both_directions_byte": 50e9,
            "latency_second": 8.92e-6,
            "flit_size_byte": 16,
            "header_size_byte": 16,
            "max_payload_size_byte": 256
        },
        "link_count_per_device": 12,
        "topology": "FC" # currently support FC (fully-connected) and RING
    },
    "device": {
        "frequency_Hz": 1410e6,
        "compute_chiplet_count": 1,
        "compute_chiplet": {
            "physical_core_count": 128, # used for area model
            "core_count": 128, # used for performance model
            "process_node": "7nm", # currently support 7nm, 6nm, 5nm
            "core": {
                "sublane_count": 4,
                "systolic_array": {
                    "array_width": 16,
                    "array_height": 16,
                    "data_type": "fp16",
                    "mac_per_cycle": 1
                },
                "vector_unit": {
                    "vector_width": 32,
                    "flop_per_cycle": 4, # 32*4=128 flops per cycle per vector unit
                    "data_type": "fp16",
                    "int32_count": 16, # the number of int32 ALUs, used for area model
                    "fp16_count": 0,
                    "fp32_count": 16,
                    "fp64_count": 8
                },
                "register_file": {
                    "num_reg_files": 1,
                    "num_registers": 16384,
                    "register_bitwidth":32,
                    "num_rdwr_ports":4
                },
                "SRAM_KB": 192
            }
        },
        "memory_protocol": "HBM2e",
        "_memory_protocol_list": [
            "HBM2e",
            "DDR4",
            "DDR5",
            "PCIe4",
            "PCIe5"
        ],
        "io": {
            "process_node": "7nm",
            "global_buffer_MB": 48,
            "physical_global_buffer_MB": 48,
            "global_buffer_bandwidth_per_cycle_byte": 5120,
            "memory_channel_physical_count": 6, # used for area model
            "memory_channel_active_count": 5, # used for performance model
            "pin_count_per_channel": 1024,
            "bandwidth_per_pin_bit": 3.2e9
        },
        "memory": {
            "total_capacity_GB": 80
        }
    }
}

```

## Step 2: Build a LLM Computational Graph

Transformer blocks have been provided as in [`transformer.py`](../software_model/transformer.py), including Initial Computation (also called Prefill or Context stage) and Auto Regression (also called Decoding or Generation stage), with Tensor Parallelism support (automatically turned of if the system only has 1 device).

The user needs to provide these parameter:
* `d_model`: the hidden dimension, 12288 for GPT3
* `n_heads`: the number of heads, 96 for GPT3
* `device_count`: tensor parallelism
* `data_type`: `int8`, `fp16`, or `fp32`

### Build Your Own LLM

The user can also build their own computational graph following the [`transformer.py`](../software_model/transformer.py) example using provided operators: [`matmul`](../software_model/matmul.py), [`softmax`](../software_model/softmax.py), [`layernorm`](../software_model/layernorm.py), [`gelu`](../software_model/gelu.py), and [`allreduce`](../software_model/communication_primitives.py).

The user needs to define a new `class` by inheriting `Operator` the class and configure these fields:
* `__init__`: define the needed operators in the initial function
* `__call__`: build the computational graph. The shape of Tensors will be automatically calculated and used for simulation.
* `compile_and_simulate`: simulate all the operators and get the total latency as well as other runtimes.
* `roofline_model` (optional): a roofline model analysis.
* `run_on_gpu` (optional): run the computational graph on real-world GPUs with PyTorch.

## Step 3: Run a LLMCompass Simulation

First, read the hardware configuration and parse it to LLMCompass:
```python
from design_space_exploration.dse import template_to_system, read_architecture_template

specs = read_architecture_template("PATH/TO/YOUR/JSON")
system = template_to_system(specs)

```

Next, initiate and instantiate an LLM as in this example:
```python
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

```

Finally, run the simulation
```
auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
	system, "heuristic-GPU"
)
```
