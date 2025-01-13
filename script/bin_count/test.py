import torch

num_embeddings = 16
b = 2
n = 40
indices = torch.randint(0, num_embeddings, (b, n)) - 10

print(indices)
indices = indices[indices >= 0]
indices = indices[indices < num_embeddings]
print(indices)
index_count = torch.bincount(indices.view(-1), minlength=num_embeddings)

print(index_count)