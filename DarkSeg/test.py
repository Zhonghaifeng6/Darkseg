import torch


a = torch.randn([1,2,4,4])

b = a.view(1,32,1,1)
print(b.shape)