import torch

path = "/media/k4_nas/disk1/Datasets/Music_Part1/Boris Bukowski/100 Stunden am Tag/01.Boris Bukowski - Trag meine Liebe wie einen Mantel_pitch.pt"
data = torch.load(path, weights_only=False)
print(data)
