import itertools

def generate_combinations(input_dict):
    # Extract keys and values from the input dictionary
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Generate the Cartesian product of the values
    combinations = list(itertools.product(*values))

    # Create a list of dictionaries from the combinations
    result = [{keys[i]: combo[i] for i in range(len(keys))} for combo in combinations]

    return result

# 示例1
input_dict1 = {"a": [1, 2], "b": [3, 4]}
combinations1 = generate_combinations(input_dict1)
for item in combinations1:
    print(item)

# 示例2
input_dict2 = {"x": [5, 6, 7], "y": ["a", "b"]}
combinations2 = generate_combinations(input_dict2)
for item in combinations2:
    print(item)

# 示例3
input_dict3 = {"p": [10], "q": [20, 30], "r": [40, 50]}
combinations3 = generate_combinations(input_dict3)
for item in combinations3:
    print(item)