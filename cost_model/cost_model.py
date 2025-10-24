# Author: August Ning aning@princeton.edu
# Date started: 12 October 2023
# This file is the cost model for Naivesim

import numpy as np
import math
# import supply_chain.supply_chain_model as scm
import cost_model.supply_chain.supply_chain_model as scm

# lots of parameters required for calculating silicon die area cost

# these are in terms of million transistors per mm2
transistor_density_7nm = scm.transistor_density_arr[scm.PN_7_INDEX]
transistor_density_6nm = 114.2
transistor_density_5nm = scm.transistor_density_arr[scm.PN_5_INDEX]

sram_bit_cell_density_7nm = 1.70e-07
sram_bit_cell_density_6nm = 1.40e-07
sram_bit_cell_density_5nm = 1.25e-07

# cache size overheads derived from cacti for cache sizes
# 4096, 8192, 16384, ..., 1 MB
cache_area_efficiency_arr = [0.076, 0.142, 0.247, 0.393, 0.559, \
                             0.704, 0.526, 0.602, 0.561]

# fpu transistor counts are for 64 bit FPU, based off Ariane and OpenPiton's SPARC T1
# assume that fp32 are half the transistors
# int32 transistor count is based off of Ariane's Mult and OpenPiton's SPARC T1
# systolic array is for 1x1 area

# scale FPU area by mantissa bits quadratically
fpu64_transistor_count = 685300
fpu32_transistor_count = fpu64_transistor_count * ((23 / 52) ** 2)
fpu16_transistor_count = fpu64_transistor_count * ((10 / 52) ** 2)

int32_transistor_count = 177690

# based off of A100 SM and MI 210 CU
# these overheads are per sublane, per vector width
# (ex 32 for A100, 16 for MI 210)
per_sublane_control_transistor_count = 996200
nvidia_per_sublane_control_transistor_count = 725650
amd_per_sublane_control_transistor_count = 1534500

per_sublane_control_dict = {'nvidia':per_sublane_control_transistor_count, \
                            'amd':per_sublane_control_transistor_count}

per_core_comm_transistor_count = 44300000
nvidia_per_core_comm_transistor_count = 55000000
amd_per_core_comm_transistor_count = 33600000

per_core_comm_dict = {'nvidia':per_core_comm_transistor_count, \
                      'amd':per_core_comm_transistor_count}


# memory controllers scale with process node, but PHYs do not
# pcie, ddr, hbm
# note: DDR link unit is 32 bits
pcie5_phy_mm2_per_lane = 0.64
pcie4_phy_mm2_per_lane = 0.48
ddr5_phy_mm2_per_link_unit = 1.45
hbm2e_phy_mm2_per_link_unit = 10.45
nvlink3_phy_mm2_per_link_unit = 1.888
nvlink4_phy_mm2_per_link_unit = 0.965
infinity_fabric_phy_mm2_per_link_unit = 5.69

pcie5_ctrl_transistors_per_lane = 5372100
pcie4_ctrl_transistors_per_lane = 3962500
ddr5_ctrl_transistors_per_link_unit = 90446400
hbm2e_ctrl_transistors_per_link_unit = 552743000
nvlink3_ctrl_transistors_per_link_unit = 74632000
nvlink4_ctrl_transistors_per_link_unit = 86628000
infinity_fabric_ctrl_transistors_per_link_unit = 348148000

# mem tech keywords
PCIE5 = 'PCIe5'
PCIE4 = 'PCIe4'
DDR5 = 'DDR5'
HBM = 'HBM2e'
NVLINK3 = 'NVLink3'
NVLINK4 = 'NVLink4'
INFINITYFABRIC = 'InfinityFabric'

# average via dramexchange spot price, Oct 2023
ddr5_cost_per_gb = 2.4
hbm_cost_per_gb = 7

# return die area for a dimension x dimension SA with a
# give bitwidth FPU at a given process node
# right now, we model each PE's MAC as a FPU
def calc_systolic_array_area_mm2(dimension_x, dimension_y, bitwidth, transistor_density_mil_mm2):
    if bitwidth == 'fp64':
        total_transistor_count = fpu64_transistor_count * dimension_x * dimension_y
    elif bitwidth == 'fp32':
        total_transistor_count = fpu32_transistor_count * dimension_x * dimension_y
    elif bitwidth == 'fp16':
        total_transistor_count = fpu16_transistor_count * dimension_x * dimension_y

    return total_transistor_count / 1e6 / transistor_density_mil_mm2

# vector width corresponds to number of FPUs you have
def calc_vector_area_mm2(int32_count, fp16_count, fp32_count, fp64_count, transistor_density_mil_mm2):
    total_transistor_count = 0
    total_transistor_count += int32_count * int32_transistor_count
    total_transistor_count += fp16_count * fpu16_transistor_count
    total_transistor_count += fp32_count * fpu32_transistor_count
    total_transistor_count += fp64_count * fpu64_transistor_count
    
    return total_transistor_count / 1e6 / transistor_density_mil_mm2

# for cache designs, if the desired capacity is larger than the max cache unit
# split them up into multiple units of the max capacity
# min cache size is 4096 bytes
def calc_cache_sram_area_mm2(capacity_bytes, sram_bitcell_area_mm2, max_cache_unit_bytes=(2**19)):
    if capacity_bytes > max_cache_unit_bytes:
        num_cache_units = math.ceil(capacity_bytes / max_cache_unit_bytes)
        unit_size_bytes = max_cache_unit_bytes
    else:
        num_cache_units = 1
        unit_size_bytes = capacity_bytes

    # cache size model is for capacity of 4096 bytes to 1 MB
    if unit_size_bytes < 2 ** 12:
        unit_size_bytes = 2 ** 12

    area_efficiency_index = math.ceil(math.log(unit_size_bytes, 2)) - 12
    area_efficiency_factor = cache_area_efficiency_arr[area_efficiency_index]
    unit_cache_area = unit_size_bytes * 8 * sram_bitcell_area_mm2 / area_efficiency_factor
    cache_area = num_cache_units * unit_cache_area
    return cache_area

# area model comes from EMPIRE
# num_reg_files: how many distinct register files each sublanes has
# D: how many registers there are in each RF
# W: bits per register
# P: number of read/write ports
def calc_reg_file_area(num_reg_files, D, W, P, transistor_density_mil_mm2):
    area_90nm_um2 = (3.29 * 10**4) - (1.09 * 10**3 * D) - (8.83 * 10**2 * W) - (5.55 * 10**3 * P) \
           + (5.35 * 10**1 * D * W) + (1.50 * 10**-2 * D**2) + (1.08 * 10**-2 * W**2) \
           + (5.86 * 10**-1 * P**2) + (1.42 * 10**2 * D * P) + (3.68 * 10**2 * W * P)
    
    # need to convert um2 to mm2, convert to 7nm
    area_90nm_mm2 = area_90nm_um2 / 1e6
    area_mm2 =  area_90nm_mm2 * (scm.transistor_density_arr[scm.PN_90_INDEX] / transistor_density_mil_mm2)
    total_reg_file_area = num_reg_files * area_mm2
    return total_reg_file_area

# for width, for PCIe and NVLink, it is the whole lane
# for DDR and HBM, it's 128 bits and 1024 bits respectively
def calc_mem_controller_area_mm2(mem_tech, width, transistor_density_mil_mm2):
    controller_transistor_count = -1

    if mem_tech == PCIE5:
        controller_transistor_count = pcie5_ctrl_transistors_per_lane * width
    elif mem_tech == PCIE4:
        controller_transistor_count = pcie4_ctrl_transistors_per_lane * width
    elif mem_tech == DDR5:
        controller_transistor_count = ddr5_ctrl_transistors_per_link_unit * width
    elif mem_tech == HBM:
        controller_transistor_count = hbm2e_ctrl_transistors_per_link_unit * width
    elif mem_tech == NVLINK3:
        controller_transistor_count = nvlink3_ctrl_transistors_per_link_unit * width
    elif mem_tech == NVLINK4:
        controller_transistor_count = nvlink4_ctrl_transistors_per_link_unit * width
    elif mem_tech == INFINITYFABRIC:
        controller_transistor_count = infinity_fabric_ctrl_transistors_per_link_unit * width

    return (controller_transistor_count / 1e6) / transistor_density_mil_mm2

def calc_mem_phy_area_mm2(mem_tech, width):
    if mem_tech == PCIE5:
        return pcie5_phy_mm2_per_lane * width
    elif mem_tech == PCIE4:
        return pcie4_phy_mm2_per_lane * width
    elif mem_tech == DDR5:
        return ddr5_phy_mm2_per_link_unit * width
    elif mem_tech == HBM:
        return hbm2e_phy_mm2_per_link_unit * width
    elif mem_tech == NVLINK3:
        return nvlink3_phy_mm2_per_link_unit * width
    elif mem_tech == NVLINK4:
        return nvlink4_phy_mm2_per_link_unit * width
    elif mem_tech == INFINITYFABRIC:
        return infinity_fabric_phy_mm2_per_link_unit * width
    else:
        return -1

def find_logic_sram_transistor_density(process_node):
    if '7' in process_node:
        return transistor_density_7nm, sram_bit_cell_density_7nm
    elif '6' in process_node:
        return transistor_density_6nm, sram_bit_cell_density_6nm
    elif '5' in process_node:
        return transistor_density_5nm, sram_bit_cell_density_5nm

    raise Exception("Invalid Process Node")


# a compute core consists of a fixed control overhead
# a specified width fp32 vector engine
# a specified dimmension fp16 systolic array
# a specified L1 cache
# at a specified process node
# NB: you can fit multiple cores onto a single die for chiplet systems
def calc_compute_chiplet_area_mm2(configs_dict, verbose=False):
    total_die_map = {}
    core_breakdown_map = {}
    device_name = configs_dict['name']
    device_brand = 'nvidia' if 'nvidia' in device_name.lower() else 'amd'
    vector_width = configs_dict['device']['compute_chiplet']['core']['vector_unit']['vector_width']
    vector_int32_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['int32_count']
    vector_fp16_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['fp16_count']
    vector_fp32_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['fp32_count']
    vector_fp64_count = configs_dict['device']['compute_chiplet']['core']['vector_unit']['fp64_count']
    sa_dim_x = configs_dict['device']['compute_chiplet']['core']['systolic_array']['array_width']
    sa_dim_y = configs_dict['device']['compute_chiplet']['core']['systolic_array']['array_height']
    sa_bitwidth = configs_dict['device']['compute_chiplet']['core']['systolic_array']['data_type']
    num_reg_files = configs_dict['device']['compute_chiplet']['core']['register_file']['num_reg_files']
    num_registers = configs_dict['device']['compute_chiplet']['core']['register_file']['num_registers']
    register_bitwidth = configs_dict['device']['compute_chiplet']['core']['register_file']['register_bitwidth']
    num_rdwr_ports = configs_dict['device']['compute_chiplet']['core']['register_file']['num_rdwr_ports']
    sublane_count = configs_dict['device']['compute_chiplet']['core']['sublane_count']
    cache_size_bytes = configs_dict['device']['compute_chiplet']['core']['SRAM_KB'] * (2 ** 10)
    process_node = configs_dict['device']['compute_chiplet']['process_node']
    cores_per_chiplet = configs_dict['device']['compute_chiplet']['physical_core_count']

    # each sublane has a SA and vector unit. a core is made up of sublanes. a chiplet has multiple cores
    transistor_density_mil_mm2, sram_density_bitcell_mm2 = find_logic_sram_transistor_density(process_node)
    per_sublane_area_mm2 = 0
    per_sublane_control_area_mm2 = per_sublane_control_dict[device_brand] / 1e6 / transistor_density_mil_mm2
    per_sublane_area_mm2 += (vector_width * per_sublane_control_area_mm2)
    control_logic_area = per_sublane_area_mm2 * sublane_count

    per_lane_vector_area = calc_vector_area_mm2(vector_int32_count, vector_fp16_count, vector_fp32_count, vector_fp64_count, transistor_density_mil_mm2)
    per_sublane_area_mm2 += per_lane_vector_area

    per_lane_sa_area = calc_systolic_array_area_mm2(sa_dim_x, sa_dim_y, sa_bitwidth, transistor_density_mil_mm2)
    per_sublane_area_mm2 += per_lane_sa_area

    per_lane_regfile_area = calc_reg_file_area(num_reg_files, num_registers, register_bitwidth, num_rdwr_ports, transistor_density_mil_mm2)
    per_sublane_area_mm2 += per_lane_regfile_area

    per_core_compute_area_mm2 = per_sublane_area_mm2 * sublane_count
    cache_area_mm2 = calc_cache_sram_area_mm2(cache_size_bytes, sram_density_bitcell_mm2)
    per_core_area_mm2 = per_core_compute_area_mm2 + cache_area_mm2

    core_breakdown_map['total_core_area'] = per_core_area_mm2
    core_breakdown_map['control_area'] = control_logic_area
    core_breakdown_map['alu_area'] = per_lane_vector_area * sublane_count
    core_breakdown_map['sa_area'] = per_lane_sa_area * sublane_count
    core_breakdown_map['regfile_area'] = per_lane_regfile_area * sublane_count
    core_breakdown_map['local_buffer_area'] = cache_area_mm2

    total_cores_area = per_core_area_mm2 * cores_per_chiplet
    total_crossbar_area = (per_core_comm_dict[device_brand] / 1e6 / transistor_density_mil_mm2) * cores_per_chiplet
    # each core has an area overhead to connect to the xbar
    compute_chiplet_area_mm2 = total_cores_area + total_crossbar_area

    total_die_map['total_area'] = compute_chiplet_area_mm2
    total_die_map['cores_area'] = total_cores_area
    total_die_map['crossbar_area'] = total_crossbar_area
    if verbose:
        return compute_chiplet_area_mm2, core_breakdown_map, total_die_map
    else:
        return compute_chiplet_area_mm2


# NB: for mem_tech, if you are using DDR or HBM, it will be 128 bits and 1024 bits respectively per lane
# for PCIe and NVLink, specify the number of lanes (128bits per lane)
# def calc_io_die_area_mm2(cache_size_bytes, mem_tech, mem_tech_width, num_nvlink_phys, \
#                          transistor_density_mil_mm2, sram_density_bitcell_mm2):
def calc_io_die_area_mm2(config_dict, verbose=False):
    total_die_map = {}

    cache_size_bytes = config_dict['device']['io']['physical_global_buffer_MB'] * (2 ** 20)
    mem_tech = config_dict['device']['memory_protocol']
    num_mem_tech_units = config_dict['device']['io']['memory_channel_physical_count']
    gpu_gpu_comm_tech = config_dict['interconnect']['link']['name']
    num_gpu_gpu_comm_phy = config_dict['interconnect']['link_count_per_device']
    process_node = config_dict['device']['io']['process_node']

    transistor_density_mil_mm2, sram_density_bitcell_mm2 = find_logic_sram_transistor_density(process_node)
    io_die_area_mm2 = 0
    io_die_area_mm2 += calc_cache_sram_area_mm2(cache_size_bytes, sram_density_bitcell_mm2)
    global_buffer_area = io_die_area_mm2

    # mem tech for communicating to off chip memory
    mem_phy_area = calc_mem_phy_area_mm2(mem_tech, num_mem_tech_units)
    mem_controller_area = calc_mem_controller_area_mm2(mem_tech, num_mem_tech_units, transistor_density_mil_mm2)
    io_die_area_mm2 += mem_phy_area
    io_die_area_mm2 += mem_controller_area

    # every  IO die has a few NV links for chip to chip communication
    device_phy_area = calc_mem_phy_area_mm2(gpu_gpu_comm_tech, num_gpu_gpu_comm_phy)
    device_controller_area = calc_mem_controller_area_mm2(gpu_gpu_comm_tech, num_gpu_gpu_comm_phy, transistor_density_mil_mm2)
    io_die_area_mm2 += device_phy_area
    io_die_area_mm2 += device_controller_area

    total_die_map['total_die_area'] = io_die_area_mm2
    total_die_map['global_buffer_area'] = global_buffer_area
    total_die_map['mem_phy_area'] = mem_phy_area
    total_die_map['mem_controller_area'] = mem_controller_area
    total_die_map['device_phy_area'] = device_phy_area
    total_die_map['device_controller_area'] = device_controller_area

    if verbose:
        return io_die_area_mm2, total_die_map
    else:
        return io_die_area_mm2
