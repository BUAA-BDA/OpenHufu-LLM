from datasets import load_dataset, interleave_datasets, load_from_disk
from collections import defaultdict
import json
import os

# 加载 COIG-CQIA 数据集
dataset1 = load_dataset("m-a-p/COIG-CQIA", 'xhs', split='train')
dataset2 = load_dataset("m-a-p/COIG-CQIA", 'chinese_traditional', split='train')
dataset3 = load_dataset("m-a-p/COIG-CQIA", 'douban', split='train')
dataset = interleave_datasets([dataset1, dataset2, dataset3])
print(dataset)
# 获取训练集
train_data = dataset

# 提取 domain 字段的最后一个域
last_domains = [domain[-1] for domain in train_data["domain"]]

# 获取所有不同的最后一个域
unique_last_domains = list(set(last_domains))

# 创建一个字典来存储每个 domain 的数据集
domain_datasets = {domain: [] for domain in unique_last_domains}

# 将数据集按照 domain 分类
for domain, data in zip(last_domains, train_data):
    domain_datasets[domain].append(data)

# 将每个 domain 的数据集转换为 Dataset
from datasets import Dataset

for domain in unique_last_domains:
    domain_datasets[domain] = Dataset.from_list(domain_datasets[domain])

# 打印每个 domain 的数据集大小
for domain, ds in domain_datasets.items():
    print(f"Domain: {domain}, Number of samples: {len(ds)}")

# 合并所有数据集
all_data = []
for domain in unique_last_domains:
    all_data.extend(domain_datasets[domain])

num_clients = 1000
client_datasets = []
client_size = len(all_data) // num_clients

for i in range(num_clients):
    start = i * client_size
    end = (i + 1) * client_size if i != num_clients - 1 else len(all_data)
    client_datasets.append(all_data[start:end])

# 统计每个客户端中每种 domain 的数据数量
client_domain_counts = defaultdict(lambda: defaultdict(int))
for client_idx, client_data in enumerate(client_datasets):
    for data in client_data:
        domain = data["domain"][-1]  # 获取 domain 的最后一个值
        client_domain_counts[client_idx][domain] += 1

# 打印结果
for client_idx in range(num_clients):
    print(f"客户端 {client_idx + 1} 的 domain 数据数量：")
    for domain, count in client_domain_counts[client_idx].items():
        print(f"  {domain}: {count} 个数据")
    print()

result = {
    "domain_sizes": {domain: len(ds) for domain, ds in domain_datasets.items()},
    "client_domain_counts": {str(client_idx): dict(counts) for client_idx, counts in client_domain_counts.items()}
}

with open("federated_learning_statistics.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("统计结果已保存到 federated_learning_statistics.json 文件中")

client_datasets_as_dataset = []
for client_data in client_datasets:
    client_datasets_as_dataset.append(Dataset.from_list(client_data))

save_dir = "client_datasets"
os.makedirs(save_dir, exist_ok=True)

test_size = 0.1
# 分割测试集和训练集
for i, client_ds in enumerate(client_datasets_as_dataset):
 # 计算测试集的大小
    test_size_abs = max(1, int(len(client_ds) * test_size))  # 确保至少有一个样本
    train_size_abs = len(client_ds) - test_size_abs

    # 分割数据集
    train_ds, test_ds = client_ds.train_test_split(test_size=test_size_abs, train_size=train_size_abs, shuffle=True, seed=42).values()

    # 保存测试集
    client_test_dir = os.path.join(save_dir, f"client_{i + 1}_test")
    test_ds.save_to_disk(client_test_dir)
    print(f"客户端 {i + 1} 的测试集已保存到 {client_test_dir}")

    # 保存训练集（可选）
    client_train_dir = os.path.join(save_dir, f"client_{i + 1}_train")
    train_ds.save_to_disk(client_train_dir)
    print(f"客户端 {i + 1} 的训练集已保存到 {client_train_dir}")

# 验证：加载保存的测试集
for i in range(len(client_datasets_as_dataset)):
    client_test_dir = os.path.join(save_dir, f"client_{i + 1}_test")
    loaded_test_ds = load_from_disk(client_test_dir)
    print(f"客户端 {i + 1} 的测试集加载成功，大小：{len(loaded_test_ds)}")