from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

# class WeightNorm(nn.Module):
# #     def __init__(self):
# #         super(WeightNorm, self).__init__()
    
#     def forward(self, input):
#         g = torch.Tensor([1]).cuda()
#         input = input/g
#         norm = input.data.norm(p=2, keepdim=True)        
#         return g * input.div(norm)

class ConvNet(nn.Module):
    def __init__(self, WordDict, Hyperparams):
        ###
        super(ConvNet, self).__init__()
        self.WordDict = WordDict
        self.EmbeddingSize = Hyperparams["EmbeddingSize"]
        self.LearningRate = Hyperparams["LearningRate"]
        self.ChannelSize = Hyperparams["ChannelSize"]
        self.WordVectorNorm = Hyperparams["WordVectorNorm"]
        self.UsePreWordVector = Hyperparams["UsePreWordVector"]
        self.BatchSize = Hyperparams["BatchSize"]
        self.KernelSize = Hyperparams["KernelSize"]
        self.MaxSeqLen = Hyperparams["MaxSeqLen"]
        self.NumClass = Hyperparams["NumClass"]
        ###
        self.embedding = nn.Embedding(num_embeddings=len(WordDict), embedding_dim=self.EmbeddingSize)
        
        if self.UsePreWordVector:
            print("Loading Pretrained Word Vectors ... ")
            PreEmbedding = np.random.randn(len(self.WordDict), self.EmbeddingSize) * 1 + 0 # N(mu, sigma)
            FileObject = open(self.UsePreWordVector, 'r')

            cnt = 0
            for line in FileObject:
                line = line.split()
                word = line[0]
                if word in self.WordDict:
                    cnt = cnt+1
                    PreEmbedding[self.WordDict[word]] = line[1:]
            self.embedding.weight.data = torch.Tensor(PreEmbedding)
            print(cnt, "Word Vectors Loaded . ")
        
        
        self.conv10 = nn.Sequential(
                nn.Conv1d(self.EmbeddingSize, self.ChannelSize[0], self.KernelSize[0]),
                nn.ReLU(),
                nn.MaxPool1d(self.MaxSeqLen-self.KernelSize[0]+1),
        )
        self.conv11 = nn.Sequential(
                nn.Conv1d(self.EmbeddingSize, self.ChannelSize[0], self.KernelSize[1]),
                nn.ReLU(),
                nn.MaxPool1d(self.MaxSeqLen-self.KernelSize[1]+1),
        )
        self.conv12 = nn.Sequential(
                nn.Conv1d(self.EmbeddingSize, self.ChannelSize[0], self.KernelSize[2]),
                nn.ReLU(),
                nn.MaxPool1d(self.MaxSeqLen-self.KernelSize[2]+1),
        )
        self.conv13 = nn.Sequential(
                nn.Conv1d(self.EmbeddingSize, self.ChannelSize[0], self.KernelSize[3]),
                nn.ReLU(),
                nn.MaxPool1d(self.MaxSeqLen-self.KernelSize[3]+1),
        )

        ### 2nd Conv
        self.conv20 = nn.Sequential(
                nn.Conv1d(len(self.KernelSize), self.ChannelSize[1], self.KernelSize[0]),
                nn.ReLU(),
                nn.MaxPool1d(self.ChannelSize[0]-self.KernelSize[0]+1),
        )
        self.conv21 = nn.Sequential(
                nn.Conv1d(len(self.KernelSize), self.ChannelSize[1], self.KernelSize[1]),
                nn.ReLU(),
                nn.MaxPool1d(self.ChannelSize[0]-self.KernelSize[1]+1),
        )
        self.conv22 = nn.Sequential(
                nn.Conv1d(len(self.KernelSize), self.ChannelSize[1], self.KernelSize[2]),
                nn.ReLU(),
                nn.MaxPool1d(self.ChannelSize[0]-self.KernelSize[2]+1),
        )
        self.conv23 = nn.Sequential(
                nn.Conv1d(len(self.KernelSize), self.ChannelSize[1], self.KernelSize[3]),
                nn.ReLU(),
                nn.MaxPool1d(self.ChannelSize[0]-self.KernelSize[3]+1),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.ChannelSize[1]*len(self.KernelSize), self.NumClass),
        )
        
    def forward(self, x):
        embed = self.embedding(x)
        embed_ = embed.transpose(1,2)
        
        out1 = self.conv10(embed_); out2 = self.conv11(embed_); out3 = self.conv12(embed_); out4 = self.conv13(embed_)
        out1 = out1.transpose(1,2); out2 = out2.transpose(1,2); out3 = out3.transpose(1,2); out4 = out4.transpose(1,2)
        out = torch.cat((out1, out2, out3, out4), 1)
        
        out1 = self.conv20(out); out2 = self.conv21(out); out3 = self.conv22(out); out4 = self.conv23(out)
        out1 = out1.transpose(1,2); out2 = out2.transpose(1,2); out3 = out3.transpose(1,2); out4 = out4.transpose(1,2)
        out = torch.cat((out1, out2, out3, out4), 1)
        
        out = out.reshape(out.size(0), -1)
        out_ = self.fc(out)
        
        return out_