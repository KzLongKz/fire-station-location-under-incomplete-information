import numpy as np


np.random.seed()


num_groups = 10
num_samples = 1000


group_means_target = np.zeros(num_groups)
group_stds_target = np.zeros(num_groups)
data = [None] * num_groups


group_means_target = np.random.uniform(41, 51, num_groups)

group_variances_target = np.random.uniform(0.81, 3.2, num_groups)

group_stds_target = np.sqrt(group_variances_target)


for i in range(num_groups):
    data[i] = np.random.normal(group_means_target[i], group_stds_target[i], num_samples)


group_means_actual = np.zeros(num_groups)
group_stds_actual = np.zeros(num_groups)

for i in range(num_groups):
    group_means_actual[i] = np.mean(data[i])
    group_stds_actual[i] = np.std(data[i], ddof=1)


print('第一次分组统计：')
print('组号\t目标均值\t实际均值\t目标标准差\t实际标准差')
for i in range(num_groups):
    print(
        f'{i + 1}\t{group_means_target[i]:.4f}\t\t{group_means_actual[i]:.4f}\t\t{group_stds_target[i]:.4f}\t\t{group_stds_actual[i]:.4f}')


subgroup_size = 100
overlap_size = 0
window_step = subgroup_size - overlap_size


min_variances = np.zeros(num_groups)
max_variances = np.zeros(num_groups)
num_subgroups = np.zeros(num_groups, dtype=int)

print(f'\n第二次分组统计（每组分成多个小组，每组{subgroup_size}个数据，重叠{overlap_size}个数据）：')


for group_idx in range(num_groups):
    current_data = data[group_idx]
    current_length = len(current_data)


    num_subgroups[group_idx] = (current_length - subgroup_size) // window_step + 1


    subgroup_vars = np.zeros(num_subgroups[group_idx])


    for subgroup_idx in range(num_subgroups[group_idx]):

        start_idx = subgroup_idx * window_step
        end_idx = start_idx + subgroup_size


        subgroup_data = current_data[start_idx:end_idx]


        subgroup_vars[subgroup_idx] = np.var(subgroup_data, ddof=1)


    min_idx = np.argmin(subgroup_vars)
    min_variances[group_idx] = subgroup_vars[min_idx]
    max_idx = np.argmax(subgroup_vars)
    max_variances[group_idx] = subgroup_vars[max_idx]


    print(
        f'组{group_idx + 1}: 最小方差={min_variances[group_idx]:.4f}, 最大方差={max_variances[group_idx]:.4f}, 子组数量={num_subgroups[group_idx]}')


print('\n所有组方差极值统计：')
print('组号\t最小方差\t最大方差\t子组数量')
for group_idx in range(num_groups):
    print(
        f'{group_idx + 1}\t{min_variances[group_idx]:.4f}\t\t{max_variances[group_idx]:.4f}\t\t{num_subgroups[group_idx]}')


overall_min_variance = np.min(min_variances)
overall_max_variance = np.max(max_variances)
print(f'\n总体方差范围: [{overall_min_variance:.4f}, {overall_max_variance:.4f}]')
print(f'总体最小方差: {overall_min_variance:.4f}')
print(f'总体最大方差: {overall_max_variance:.4f}')
