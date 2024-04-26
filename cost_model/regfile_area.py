def calculate_regfile_area(D, W, P):
    area_90nm_um2 = (3.29 * 10**4) - (1.09 * 10**3 * D) - (8.83 * 10**2 * W) - (5.55 * 10**3 * P) \
           + (5.35 * 10**1 * D * W) + (1.50 * 10**-2 * D**2) + (1.08 * 10**-2 * W**2) \
           + (5.86 * 10**-1 * P**2) + (1.42 * 10**2 * D * P) + (3.68 * 10**2 * W * P)
    
    # need to convert um2 to mm2, convert to 7nm
    area_90nm_mm2 = area_90nm_um2 / 1e6
    area_7nm_mm2 =  area_90nm_mm2 * (1.6 / 96.3)
    return area_7nm_mm2

reg_area = calculate_regfile_area(16384, 32, 4)
print(reg_area)

reg_area = 64 * calculate_regfile_area(512, 32, 4)
print(reg_area)

reg_area = calculate_regfile_area(800, 32, 4)
print(reg_area)