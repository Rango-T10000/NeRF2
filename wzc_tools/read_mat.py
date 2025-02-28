import scipy.io as scio

# Load the .mat file
data = scio.loadmat('logs/fsc/fsc-exp-progressive_gt_traj-3/results/result_1.mat')

# Print the keys of the loaded data
print("Keys in the .mat file:", data.keys())

# Print the content of each key
for key in data:
    if key.startswith('__'):
        continue  # Skip meta keys
    print(f"\nContent of {key}:")
    print(data[key])