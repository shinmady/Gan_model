import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        # 將標籤 (0-9) 轉換為可以與雜訊結合的特徵向量
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Tanh() # 輸出範圍至 [-1, 1]
        )

    def forward(self, noise, labels):
        # 取得標籤的 embedding
        c = self.label_embedding(labels)
        # 將雜訊與標籤特徵結合
        x = torch.cat([noise, c], 1)
        # 生成圖片
        img = self.model(x)
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        # 將標籤 (0-9) 轉換為可以與圖片結合的特徵向量
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # 輸出為真偽的機率值 [0, 1]
        )

    def forward(self, img, labels):
        # 將圖片攤平
        img_flat = img.view(img.size(0), -1)
        # 取得標籤的 embedding
        c = self.label_embedding(labels)
        # 將圖片與標籤特徵結合
        x = torch.cat([img_flat, c], 1)
        # 判斷真偽
        validity = self.model(x)
        return validity
