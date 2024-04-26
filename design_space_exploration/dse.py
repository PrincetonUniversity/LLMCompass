import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
# from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil

def read_architecture_template(file_path):
    with open(file_path, "r") as f:
        arch_specs = json.load(f)
    return arch_specs


def template_to_system(arch_specs):
    device_specs = arch_specs["device"]
    compute_chiplet_specs = device_specs["compute_chiplet"]
    io_specs = device_specs["io"]
    core_specs = compute_chiplet_specs["core"]
    sublane_count = core_specs["sublane_count"]
    # vector unit
    vector_unit_specs = core_specs["vector_unit"]
    vector_unit = VectorUnit(
        sublane_count
        * vector_unit_specs["vector_width"]
        * vector_unit_specs["flop_per_cycle"],
        int(re.search(r"(\d+)", vector_unit_specs["data_type"]).group(1)) // 8,
        35,
        vector_unit_specs["vector_width"],
        sublane_count,
    )
    # systolic array
    systolic_array_specs = core_specs["systolic_array"]
    systolic_array = SystolicArray(
        systolic_array_specs["array_height"],
        systolic_array_specs["array_width"],
        systolic_array_specs["mac_per_cycle"],
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
    )
    # core
    core = Core(
        vector_unit,
        systolic_array,
        sublane_count,
        core_specs["SRAM_KB"] * 1024,
    )
    # compute module
    compute_module = ComputeModule(
        core,
        compute_chiplet_specs["core_count"] * device_specs["compute_chiplet_count"],
        device_specs["frequency_Hz"],
        io_specs["global_buffer_MB"] * 1024 * 1024,
        io_specs["global_buffer_bandwidth_per_cycle_byte"],
        overhead_dict["A100"],
    )
    # io module
    io_module = IOModule(
        io_specs["memory_channel_active_count"]
        * io_specs["pin_count_per_channel"]
        * io_specs["bandwidth_per_pin_bit"]
        // 8,
        1e-6,
    )
    # memory module
    memory_module = MemoryModule(
        device_specs["memory"]["total_capacity_GB"] * 1024 * 1024 * 1024
    )
    # device
    device = Device(compute_module, io_module, memory_module)
    # interconnect
    interconnect_specs = arch_specs["interconnect"]
    link_specs = interconnect_specs["link"]
    link_module = LinkModule(
        link_specs["bandwidth_per_direction_byte"],
        link_specs["bandwidth_both_directions_byte"],
        link_specs["latency_second"],
        link_specs["flit_size_byte"],
        link_specs["max_payload_size_byte"],
        link_specs["header_size_byte"],
    )
    interconnect_module = InterConnectModule(
        arch_specs["device_count"],
        TopologyType.FC
        if interconnect_specs["topology"] == "FC"
        else TopologyType.RING,
        link_module,
        interconnect_specs["link_count_per_device"],
    )

    # system
    system = System(device, interconnect_module)

    return system


def test_template_to_system():
    arch_specs = read_architecture_template("configs/template.json")
    A100_system = template_to_system(arch_specs)
    bs = 8
    s = 2048
    model = TransformerBlockInitComputationTP(
        d_model=12288,
        n_heads=96,
        device_count=4,
        data_type=data_type_dict["fp16"],
    )
    _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
    model.roofline_model(A100_system)


def find_cheapest_design(
    d_model,
    n_heads,
    n_layers,
    batch_size,
    input_seq_length,
    init_latency,
    output_seq_length,
    auto_regression_latency,
    
):
    i=0
    smallest_total_area_mm2=float('inf')
    best_arch_specs=None
    arch_specs = read_architecture_template("configs/template.json")
    for device_count in [4, 8, 12, 16]:
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
                data_type=data_type_dict["fp16"],)
        _ = model_init(Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"]))
        _ = model_auto_regression(Tensor([batch_size, 1, model_init.d_model],data_type_dict["fp16"]), input_seq_length+output_seq_length)
        arch_specs["device_count"] = device_count
        if device_count <= 4:
            topology = "FC"
        else:
            topology = "RING"
        arch_specs["interconnect"]["topology"] = topology
        for link_count_per_device in [6, 12, 18, 24]:
            arch_specs["interconnect"]["link_count_per_device"] = link_count_per_device
            # device
            for core_count in [32, 64, 128, 256]:
                arch_specs["device"]["compute_chiplet"]["core_count"] = core_count
                # core
                for sublane_count in [1, 2, 4, 8]:
                    arch_specs["device"]["compute_chiplet"]["core"][
                        "sublane_count"
                    ] = sublane_count
                    # systolic array
                    for array_height in [16, 32, 64, 128]:
                        arch_specs["device"]["compute_chiplet"]["core"][
                            "systolic_array"
                        ]["array_height"] = array_height
                        arch_specs["device"]["compute_chiplet"]["core"][
                            "systolic_array"
                        ]["array_width"] = array_height
                        # vector unit
                        for vector_width in [16, 32, 64, 128]:
                            arch_specs["device"]["compute_chiplet"]["core"][
                                "vector_unit"
                            ]["vector_width"] = vector_width
                            for SRAM_KB in [64, 128, 256, 512, 1024]:
                                arch_specs["device"]["compute_chiplet"]["core"][
                                    "SRAM_KB"
                                ] = SRAM_KB
                                # global buffer
                                for total_global_buffer_MB in [
                                    80,
                                    160,
                                    240,
                                    320,
                                    400,
                                    480,
                                    640,
                                    800,
                                    960,
                                ]:
                                    global_buffer_MB = (
                                        total_global_buffer_MB // device_count
                                    )
                                    global_buffer_bandwidth_per_cycle_byte = (
                                        5120 * global_buffer_MB // 40
                                    )
                                    arch_specs["device"]["io"][
                                        "global_buffer_MB"
                                    ] = global_buffer_MB
                                    arch_specs["device"]["io"][
                                        "global_buffer_bandwidth_per_cycle_byte"
                                    ] = global_buffer_bandwidth_per_cycle_byte
                                    # memory
                                    memory_capacity_requirement_GB = ceil(model_auto_regression.memory_requirement*n_layers/1e9/16)*16
                                    # print(f"memory_capacity_requirement_GB={model_auto_regression.memory_requirement*n_layers/1e9}")
                                    # exit()
                                    for memory_protocol in [
                                        "HBM2e",
                                        "DDR5",
                                        "PCIe5",
                                        # "GDDR6X"
                                    ]:
                                        arch_specs['device']['memory_protocol']=memory_protocol
                                        if memory_protocol == "HBM2e":
                                            # 400 GB/s per channel, 16 GB
                                            channel_count=memory_capacity_requirement_GB // 16
                                            if channel_count>8:
                                                continue
                                            channel_count_list = [channel_count]
                                            pin_count_per_channel=1024
                                            bandwidth_per_pin_bit=3.2e9
                                        elif memory_protocol == "DDR5":
                                            # 19.2 GB/s per channel, 2 channel per dimm
                                            channel_count_list = [16, 24, 32]
                                            pin_count_per_channel=32
                                            bandwidth_per_pin_bit=4.8e9
                                        elif memory_protocol == "PCIe5":
                                            # 4 GB/s per channel
                                            channel_count_list = [64, 96, 128]
                                            pin_count_per_channel=1
                                            bandwidth_per_pin_bit=32e9
                                        # elif memory_protocol == "GDDR6X":
                                        #     # 84 GB/s per channel, 2 GB
                                        #     channel_count_list= memo
                                        for channel_count in channel_count_list:
                                            arch_specs['device']['memory']['total_capacity_GB'] = memory_capacity_requirement_GB
                                            arch_specs['device']['io']['memory_channel_active_count'] = channel_count
                                            arch_specs['device']['io']['memory_channel_physical_count'] = channel_count
                                            arch_specs['device']['io']['pin_count_per_channel'] = pin_count_per_channel
                                            arch_specs['device']['io']['bandwidth_per_pin_bit'] = bandwidth_per_pin_bit
                                            
                                            total_area_mm2=calc_compute_chiplet_area_mm2(arch_specs)+calc_io_die_area_mm2(arch_specs)
                                            # print(f"channel_count={arch_specs['device']['io']['memory_channel_active_count']},total area={total_area_mm2}")
                                            if total_area_mm2>900:
                                                continue
                                            system=template_to_system(arch_specs)
                                            init_roofline_latency=model_init.roofline_model(system)*n_layers
                                            if init_roofline_latency>init_latency:
                                                continue
                        
                                            auto_regression_roofline_latency=model_auto_regression.roofline_model(system)*n_layers
                                            if auto_regression_roofline_latency>auto_regression_latency:
                                                continue
                                            auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(system, 'heuristic-GPU')
                                            if auto_regression_latency_simulated>auto_regression_latency:
                                                continue
                                            init_latency_simulated = model_init.compile_and_simulate(system, 'heuristic-GPU')
                                            if init_latency_simulated>init_latency:
                                                continue
                                            if total_area_mm2*device_count<smallest_total_area_mm2:
                                                smallest_total_area_mm2=total_area_mm2*device_count
                                                best_arch_specs=arch_specs
                                                best_arch_specs['area_per_device_mm2']=total_area_mm2
                                                # print(f"best_arch_specs={best_arch_specs}")
                                                # print(f"smallest_total_area_mm2={smallest_total_area_mm2}")
                                            i=i+1
                                            if i%100==0:
                                                print(f'i={i}')
    print(f'number of potential designs={i}')
    with open("configs/best_arch_specs.json", "w") as f:
        json.dump(best_arch_specs, f, indent=4)
                                            

if __name__ == "__main__":
    # test_template_to_system()
    find_cheapest_design(12288, 96, 96, 8, 2048, 5, 1024, 0.1)
    
    
