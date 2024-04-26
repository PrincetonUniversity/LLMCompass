################################################################################
# BSD 3-Clause License
#
# Copyright (c) 2023, Princeton University
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################

import numpy as np
import math

process_nodes = np.array([250, 180, 130, 90, 65, 40, 28, 20, 14, 10, 7, 5])
process_nodes_index = np.array([i for i in range(len(process_nodes))])

PN_250_INDEX = 0
PN_180_INDEX = 1
PN_130_INDEX = 2
PN_90_INDEX = 3
PN_65_INDEX = 4
PN_40_INDEX = 5
PN_28_INDEX = 6
PN_20_INDEX = 7
PN_14_INDEX = 8
PN_10_INDEX = 9
PN_7_INDEX = 10
PN_5_INDEX = 11

# process node difficulty is derived from verifcation costs and 
# drc rules count per process node
process_node_difficulty_arr = [1.7347e-08, 4.8724e-08, 9.5922e-08, \
                               1.6692e-07, 2.7371e-07, 4.3436e-07, \
                               6.7600e-07, 1.0395e-06, 1.5862e-06, \
                               2.4087e-06, 3.6458e-06, 5.5068e-06 ]

# wafer costs are taken from "AI Chips" by Saif M. Khan and Alexander Mann
# we depericate reported for the oldest process nodes from Moonwalk
# however, we assume all wafers are 300mm wafers and costs are adjusted
# according for legacy nodes
wafer_costs_arr = [900, 1020, 1580, 1650, 1937, 2274, \
                   2891, 3677, 3984, 5992, 9346, 16988]

# transistor densities are taken from "AI Chips" by Saif M. Khan and Alexander Mann
# transistor density is million transistors per mm2
transistor_density_arr = np.array([0.074, 0.39, 0.92, 1.6, 3.3, 7.7, 15.3, 22.1, 28.9, 52.5, 96.3, 138.2])
transistor_node_indices = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])

# wafer production rate is estimated from tsmc's quarterly earnings
# which report revenue and percentage of revenue per process node
# we use our wafer cost estimates to derive wafers produced per hour

# NOTE: the values here are based on TSMC's 2023 Q1 earnings report
# which is more up to date from the values presented in the paper
wafers_prod_per_hour = [8.6008e+01, 3.7945e+02, 9.7984e+01, \
                        9.3827e+01, 2.3978e+02, 2.3828e+02, \
                        3.2130e+02, 2.1052e+01, 2.5258e+02, \
                        0.0000e+00, 1.6565e+02, 1.4125e+02 ]

tsmc_wafers_per_hour = wafers_prod_per_hour

# defect density is estimated from TSMC's public slides for 10nm and 5nm
defect_density_vector_mm2 = np.array([500, 500, 500, 500, 500, 500, 500, 800, 1000, 1200, 1500, 2000]) / 1e6
# foundry latency and OSAT latency is based on disclosures 
# from Semiconductor Industry Association
base_fab_latency_weeks = np.array([12, 12, 12, 12, 12, 12, 12, 14, 14, 16, 18, 20])
base_osat_latency_weeks = 6

# the efforts are derived from verification/validation costs
# and physical and packaging costs respectively
# in practice, we found testing and packaging effort are 
# dominated by the base latency anyways, so it is the same across proccess nodes
osat_testing_effort  = 9.2e-11
osat_packaging_effort = 1.5e-5

hours_per_week = 168

######## Time-To-Market Modeling Functions ########

# scaling factor converts the curve into a reasonable number that fits the model
def process_node_difficulty(pn_i):
  return process_node_difficulty_arr[pn_i]

# unique transistor count is per transistor
# tapeout time is returned in engineering hours
def tapeout_time(num_transistors, unique_transistor_ratio, process_node_difficulty):
  unique_transistors = num_transistors
  if unique_transistor_ratio != None:
    unique_transistors = num_transistors * unique_transistor_ratio

  baseline_tapeout_eng_hours = 96
  # this should be in engineering hours
  return baseline_tapeout_eng_hours + unique_transistors * process_node_difficulty

def dies_per_wafer_area(die_area):
  wafer_diameter = 300
  wafer_area = math.pow((wafer_diameter / 2), 2) * math.pi
  dies_per_wafer = (wafer_area / die_area) - ( math.pi * (wafer_diameter) / math.sqrt(2 * die_area) )
  if dies_per_wafer < 0:
    raise Exception("die area is larger than wafer area")
  return dies_per_wafer

def num_wafers_needed(die_area, num_dies_needed):
  return num_dies_needed / dies_per_wafer_area(die_area)

# this assumes that chips with multiple dies in the same package
# are with dies that all have the same die area
def num_dies_needed(num_final_chips, dies_per_package, yield_rate):
  total_num_dies = num_final_chips * dies_per_package
  return total_num_dies / yield_rate

def yield_rate(die_area, defect_density, cluster_param):
  die_yield = math.pow( ( 1 + ( ( die_area * defect_density ) / cluster_param ) ), -cluster_param )
  return die_yield

# packaging time ratio is the assume ratio of how long it takes to test vs time to assemble
def osat_time(num_total_transistors, die_area, base_osat_time_weeks, packaging_test_ratio):
  testing_time_weeks = base_osat_time_weeks * (1-packaging_test_ratio) + \
                         + (num_total_transistors * osat_testing_effort)
  packaging_time_weeks = (base_osat_time_weeks * packaging_test_ratio) + \
                          (die_area * osat_packaging_effort)
  return testing_time_weeks + packaging_time_weeks

def fab_queue_time(wafers_per_hour, num_wafers_ahead):
  if (wafers_per_hour == 0):
    return 1e20
  return (num_wafers_ahead / wafers_per_hour ) / hours_per_week

def fab_prod_time(num_wafers_needed, wafers_per_hour, fab_and_osat_latency):
  if (wafers_per_hour == 0):
    return 1e20

  added_hours = num_wafers_needed / wafers_per_hour
  added_weeks = added_hours / hours_per_week
  return fab_and_osat_latency + added_weeks

def million_transistors_in_area_mm2(pn_i, area_mm2):
  td = transistor_density_arr[pn_i]
  return td * area_mm2

def get_die_area_mm2(num_million_transistor_per_die, transistor_density_million_mm2):
  die_area = num_million_transistor_per_die / transistor_density_million_mm2
  return die_area

######## Cost modeling (Moonwalk) ########

# ip costs are updated for advance process nodes based on design costs
mw_ip_costs_arr = [1.3500e+04, 1.3500e+04, 3.2400e+05, \
                   5.2200e+05, 6.7860e+05, 8.5020e+05, \
                   1.0335e+06, 1.6575e+06, 1.8720e+06, \
                   2.1294e+06, 2.3431e+06, 2.6442e+06 ]
# backend labor per gate is from Moonwalk and updated based on design costs
mw_backend_labor_per_gate = [0.127, 0.127, 0.127, \
                             0.127, 0.127, 0.129, \
                             0.131, 0.263, 0.280, \
                             0.340, 0.350, 0.350 ]

# moonwalk reports backend labor in terms of cost per gate
# and our metrics are per transistor count - assume 6 transistors per gate
mw_transistors_per_gate = 6

# node independent
# frontend is design time, so may not be used in our example
# cad licenses are per engineering week (are licensed per seat)
# NOTE: these costs are not updated from Moonwalk's original numbers (2017)
mw_frontend_labor_cost_per_week = 120e3/52
mw_frontend_cad_license_per_eng_week = 4e3/4
mw_backend_labor_cost_per_week = 100e3/52
mw_backend_cad_license_per_eng_week = 20e3/4
mw_flip_chip_bga_package_design_cost = 105e3

# assembly and packaging cost are  
# highly dependent on how you package
# these are starter examples
mw_assembly_cost_per_package = 1
mw_package_cost = 20

# IC knowledge strategic cost model
mw_mask_costs_arr = [4.300e+04, 7.600e+04, 1.200e+05, \
                     1.650e+05, 2.080e+05, 5.000e+05, \
                     1.200e+06, 1.900e+06, 2.820e+06, \
                     4.740e+06, 1.050e+07, 1.810e+07 ]

def tapeout_cost_weeks(pn_i, num_unique_transistors, tapeout_eng_weeks, num_engineers, tapeout_cal_weeks):
  tapeout_cost = 0
  # assume each tapeout requires baseline licensed IP costs
  tapeout_cost += mw_ip_costs_arr[pn_i]

  # add the costs of licensing for EDA tools, per engineering week
  tapeout_cost += (mw_backend_cad_license_per_eng_week * tapeout_eng_weeks)
  
  # add the cost of salaries per engineer, per calendar week, per engineer
  tapeout_cost += (mw_backend_labor_cost_per_week * num_engineers * tapeout_cal_weeks)

  # add the backend labor cost per gate, 
  tapeout_cost += (mw_backend_labor_per_gate[pn_i] * num_unique_transistors / mw_transistors_per_gate)

  # add packaging IP design cost
  tapeout_cost += mw_flip_chip_bga_package_design_cost

  return tapeout_cost

def fab_pack_cost_weeks(pn_i, num_wafers, num_packages, custom_pack_cost=None):
  fab_pack_cost = 0

  # add in the masks cost
  fab_pack_cost += mw_mask_costs_arr[pn_i]

  # add in cost of all the wafers to create the chip
  fab_pack_cost += (wafer_costs_arr[pn_i] * num_wafers)

  # add in package cost and cost of assembly (per package)
  package_cost = mw_package_cost
  if custom_pack_cost != None:
    package_cost = custom_pack_cost
  fab_pack_cost += ( num_packages * (mw_assembly_cost_per_package + package_cost) )

  return fab_pack_cost

######## TTM and CAS Functions ########
# calculates a chip's time to market (in weeks) and cost ($)
# returns a tuple of (tapeout_time, queuing_time, foundry_time, osat_time, cost)
def calc_ttm_cost_single_process(pn_i, num_final_chips, num_total_transistors, num_unique_transistors, \
             num_engineers, tapeout_parallel_factor, expected_fab_queue_weeks, wafer_per_hour_rate, dies_per_package, pack_test_ratio):
  # tapeout time is calculated in engineering hours, so need to convert to calendar weeks
  chip_tapeout_effort = process_node_difficulty(pn_i)
  chip_tapeout_eng_hours = tapeout_time(num_unique_transistors, None, chip_tapeout_effort)
  chip_tapeout_cal_weeks = (chip_tapeout_eng_hours  * (1-tapeout_parallel_factor) + \
                           (tapeout_parallel_factor * (chip_tapeout_eng_hours / num_engineers)) \
                           ) / 40
  
  # fabrication time
  chip_die_area_mm2 = get_die_area_mm2(num_total_transistors / 1e6, transistor_density_arr[pn_i])
  chip_yield_rate = yield_rate(chip_die_area_mm2, defect_density_vector_mm2[pn_i], 3)
  fab_num_dies_needed = num_dies_needed(num_final_chips, dies_per_package, chip_yield_rate)
  fab_num_wafers_needed = num_wafers_needed(chip_die_area_mm2, fab_num_dies_needed)

  # fabrication queue time. Need to convert week -> wafer -> weeks 
  # to account for when wafer prod rate changes during a disruption
  queue_time_num_wafers = expected_fab_queue_weeks * hours_per_week * wafer_per_hour_rate
  fab_queue_time_weeks = fab_queue_time(wafer_per_hour_rate, queue_time_num_wafers)

  # find the osat latency
  chip_osat_time = osat_time(num_total_transistors, chip_die_area_mm2, base_osat_latency_weeks, pack_test_ratio)
  chip_fab_osat_latency = base_fab_latency_weeks[pn_i] + chip_osat_time
  chip_fab_osat_time = fab_prod_time(fab_num_wafers_needed, wafer_per_hour_rate, chip_fab_osat_latency)

  chip_foundry_time = chip_fab_osat_time - chip_osat_time
  # find tapeout costs and fab and packaging costs
  chip_tapeout_costs = tapeout_cost_weeks(pn_i, num_unique_transistors, \
                      (chip_tapeout_eng_hours / 40), num_engineers, chip_tapeout_cal_weeks)
  chip_fab_costs = fab_pack_cost_weeks(pn_i, fab_num_wafers_needed, num_final_chips, None)

  return (chip_tapeout_cal_weeks, fab_queue_time_weeks, chip_foundry_time, \
          chip_osat_time, (chip_tapeout_costs + chip_fab_costs)) 

# given an input vectors of a chip's time to market and wafer production rates 
# (percentage of max prod rate), calculate Chip Agility Score
def calc_cas_single_process(ttm_v, wafer_prod_rate_v):
  if (len(ttm_v) != len(wafer_prod_rate_v)):
    raise Exception("TTM and wafer production rate arrays aren't the same length")
  
  ttm_diff = np.diff(ttm_v)
  prod_diff = np.diff(wafer_prod_rate_v)
  ttm_prod_roc = abs(ttm_diff / prod_diff)
  cas_v = np.power(ttm_prod_roc, -1)
  return cas_v
