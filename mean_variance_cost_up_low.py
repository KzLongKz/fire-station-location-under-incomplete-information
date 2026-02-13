import numpy as np


np.random.seed()

# 参数设置
num_groups = 10
group_size = 1000
total_samples = num_groups * group_size
target_mean = 6.5


group_variances = np.random.uniform(0, 3, [num_groups, 1])


data = []
for i in range(num_groups):

    group_data = np.random.normal(target_mean, np.sqrt(group_variances[i]), [1, group_size])
    if len(data) == 0:
        data = group_data
    else:
        data = np.concatenate((data, group_data), axis=1)


actual_mean = np.mean(data)
print(f'实际总体均值: {actual_mean:.4f}')


new_group_size = 100
overlap_size = 0
window_step = new_group_size - overlap_size
num_new_groups = int((len(data[0]) - new_group_size) / window_step) + 1

new_group_variances = np.zeros(num_new_groups)
for i in range(num_new_groups):
    start_index = i * window_step
    end_index = start_index + new_group_size
    group_data = data[0, start_index:end_index]
    new_group_variances[i] = np.var(group_data, ddof=1)


min_var = np.min(new_group_variances)
max_var = np.max(new_group_variances)
print(f'最小方差: {min_var:.4f}')
print(f'最大方差: {max_var:.4f}')


min_std = np.sqrt(min_var)
max_std = np.sqrt(max_var)
print(f'最小标准差: {min_std:.4f}')
print(f'最大标准差: {max_std:.4f}')


# print('\n10组的目标方差值:')
# print(group_variances.T)