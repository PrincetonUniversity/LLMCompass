from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from design_space_exploration.dse import read_architecture_template
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

arch_specs = read_architecture_template("configs/GA100.json")

compute_chiplet_area_mm2, A100_core_breakdown_map, compute_total_die_map = (
    calc_compute_chiplet_area_mm2(arch_specs, verbose=True)
)
io_die_area_mm2, io_total_die_map = calc_io_die_area_mm2(arch_specs, verbose=True)


categories = [
    "Cores",
    "On-chip interconnect",
    "Global buffer",
    "Memory(PHY)",
    "Memory(Control)",
    "Off-chip interconnect\n(PHY)",
    "Off-chip interconnect\n(Control)",
    "Other",
]

die_area = pd.read_csv(
    "ae/figure6/real_hardware/die_area.csv", header=None, names=["A100", "MI210"]
)

values_a100 = die_area["A100"].tolist()
values_mi210 = die_area["MI210"].tolist()

values_a100_sim = [
    compute_total_die_map["cores_area"],
    compute_total_die_map["crossbar_area"],
    io_total_die_map["global_buffer_area"],
    io_total_die_map["mem_phy_area"],
    io_total_die_map["mem_controller_area"],
    io_total_die_map["device_phy_area"],
    io_total_die_map["device_controller_area"],
    0,
]


arch_specs = read_architecture_template("configs/mi210_template.json")

compute_chiplet_area_mm2, MI210_core_breakdown_map, compute_total_die_map = (
    calc_compute_chiplet_area_mm2(arch_specs, verbose=True)
)
io_die_area_mm2, io_total_die_map = calc_io_die_area_mm2(arch_specs, verbose=True)


values_mi210_sim = [
    compute_total_die_map["cores_area"],
    compute_total_die_map["crossbar_area"],
    io_total_die_map["global_buffer_area"],
    io_total_die_map["mem_phy_area"],
    io_total_die_map["mem_controller_area"],
    io_total_die_map["device_phy_area"],
    io_total_die_map["device_controller_area"],
    0,
]

plt.figure(figsize=(4, 2))

colors_matmul = sns.color_palette("flare_r", 7)[5:6]
colors_normalization = sns.color_palette("summer", 2)
colors_gelu = sns.color_palette("pink", 5)[2:4]
colors_allreduce = sns.color_palette("Blues_r", 2)
colors = (
    colors_matmul
    + colors_normalization
    + colors_gelu
    + colors_allreduce
    + sns.color_palette("Greys_r", 1)
)

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_a100)):
    plt.bar(1, value, bottom=bottom, color=colors[i], label=category, width=0.5)
    bottom += value

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_a100_sim)):
    plt.bar(2, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_mi210)):
    plt.bar(3, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_mi210_sim)):
    plt.bar(4, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value
plt.ylabel("Area ($mm^2$)")
plt.xticks(
    [1, 2, 3, 4],
    ["Real\nGA100", "Simulated\nGA100", "Real\nAldebaran", "Simulated\nAldebaran"],
)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1.1))
plt.savefig("ae/figure6/figure6a.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


values_a100 = [3.75]
values_mi210 = [4.02]

values_a100_sim = [
    A100_core_breakdown_map["control_area"],
    A100_core_breakdown_map["alu_area"],
    A100_core_breakdown_map["sa_area"],
    A100_core_breakdown_map["regfile_area"],
    A100_core_breakdown_map["local_buffer_area"],
]

values_mi210_sim = [
    MI210_core_breakdown_map["control_area"],
    MI210_core_breakdown_map["alu_area"],
    MI210_core_breakdown_map["sa_area"],
    MI210_core_breakdown_map["regfile_area"],
    MI210_core_breakdown_map["local_buffer_area"],
]

categories = [
    "Control logic",
    "ALUs",
    "Systolic array",
    "Register file",
    "Local buffer",
]

colors = colors_matmul + colors_normalization + colors_allreduce
color_gt = sns.color_palette("Greys_r", 1)[0]

plt.figure(figsize=(4, 1.5))

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_a100)):
    plt.bar(1, value, bottom=bottom, color=color_gt, width=0.5)
    bottom += value

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_a100_sim)):
    plt.bar(2, value, bottom=bottom, color=colors[i], label=category, width=0.5)
    bottom += value

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_mi210)):
    plt.bar(3, value, bottom=bottom, color=color_gt, width=0.5)
    bottom += value

bottom = 0
for i, (category, value) in enumerate(zip(categories, values_mi210_sim)):
    plt.bar(4, value, bottom=bottom, color=colors[i], width=0.5)
    bottom += value

plt.ylabel("Area ($mm^2$)")
plt.xticks(
    [1, 2, 3, 4],
    ["Real\nGA100", "Simulated\nGA100", "Real\nAldebaran", "Simulated\nAldebaran"],
)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1))
plt.savefig("ae/figure6/figure6b.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
