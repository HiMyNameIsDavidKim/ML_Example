import torch

n = 25  # 예시로 n을 10으로 설정

for i in torch.arange(0.1, 1, 0.9/n):
    print(i)

print(len(torch.arange(0.1, 1, 0.9/n)))
print(0.9/n)
