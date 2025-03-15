import torch

B, C, H, W = 1, 1, 1, 6
input = torch.arange(6).view(B, C, H, W).float()
print(input.shape)  # torch.Size([1, 1, 1, 6])
print(input)


from torch.nn import functional as F

kernel_size = (1, 3)
stride = 2

unfold_output = F.unfold(input, kernel_size=kernel_size, stride=stride)
print(unfold_output.shape)  # torch.Size([1, 3, 2])
print(unfold_output)