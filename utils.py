from typing import List

def size_of_list(list: List):
    result = 1
    for i in list:
        result *= i
    return result

def size(list):
    if isinstance(list, List):     
        return size_of_list(list)
    else:
        return list.size

def closest_factors(n):
    x = int(n**0.5)
    while x >= 1:
        if n % x == 0:
            return x, n // x
        x -= 1
    return 0,0