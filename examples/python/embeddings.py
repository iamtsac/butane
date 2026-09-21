import torch
import butane
import matplotlib.pyplot as plt

if __name__ == "__main__":
    emb = butane.nn.embeddings.SinusoidalEmbeddings(18).to('cuda')
    positions = torch.arange(64).to('cuda')
    print(emb(positions).size())

    emb = butane.nn.embeddings.LearnableEmbeddings(18).to('cuda')
    positions = torch.arange(64).to('cuda')
    print(emb(positions).size())
