import cost_model.cost_model as cost_model
import json

# example chip with a 32 wide vector, 16x16 SA, 256kb cache core, 8 cores per die
# io die with 64 mb cache, 8 nvlinks, 32 pcie phys
# all at 5nm and 7nm

with open("./configs/prefilling_system.json", "r") as f:
    # with open('../configs/mi210_template.json', 'r') as f:
    configs_dict = json.load(f)

# print(configs_dict['device'])
# print(data['device']['compute_chiplet_count'])

compute_area = cost_model.calc_compute_chiplet_area_mm2(configs_dict)
io_area = cost_model.calc_io_die_area_mm2(configs_dict)
print(
    f"compute area: {compute_area}, io area: {io_area}, total area: {compute_area+io_area}"
)

exit(0)
core_compute_area_mm2 = cost_model.calc_compute_core_area_mm2(
    32,
    16,
    2**18,
    cost_model.transistor_density_7nm,
    cost_model.sram_bit_cell_density_7nm,
)
io_die_area_mm2 = cost_model.calc_io_die_area_mm2(
    2**25,
    cost_model.PCIE5,
    32,
    8,
    cost_model.transistor_density_7nm,
    cost_model.sram_bit_cell_density_7nm,
)
print(core_compute_area_mm2)
print(io_die_area_mm2)
