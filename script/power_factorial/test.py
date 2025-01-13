def power_factorial_ratio_optimized(x, n):
    if n < 0:
        raise ValueError("n必须是非负整数")
    
    # 特殊情况处理
    if n == 0:
        return 1
    
    # 同时计算分子和分母，避免大数计算
    result = 1
    for i in range(1, n + 1):
        result *= (x / i)
    
    return result

# print(power_factorial_ratio_optimized(2, 16))

# print(power_factorial_ratio_optimized(1, 16))

# print(power_factorial_ratio_optimized(4, 16))

def power_sqrt_factorial_ratio_optimized(x, n):
    if n < 0:
        raise ValueError("n必须是非负整数")
    
    # 特殊情况处理
    if n == 0:
        return 1
    
    # 同时计算分子和分母，避免大数计算
    result = 1
    for i in range(1, n + 1):
        result *= (x / (i ** 0.5))
    
    return result

# print(power_sqrt_factorial_ratio_optimized(2, 16))

# print(power_factorial_ratio_optimized(2, 16))

print(power_sqrt_factorial_ratio_optimized(2, 128))

print(power_factorial_ratio_optimized(2, 128))

print(power_sqrt_factorial_ratio_optimized(1, 16))
print(power_sqrt_factorial_ratio_optimized(1, 32))

print(power_sqrt_factorial_ratio_optimized(1, 64))