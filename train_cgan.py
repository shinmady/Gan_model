import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cgan_model import Generator, Discriminator
import os

# 超參數設定
latent_dim = 100
num_classes = 10
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 10 # 為了測試，預設設小一點，可自行調大 (如 50-100)

def train():
    os.makedirs("models", exist_ok=True)
    
    # 準備資料集 (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    print("正在下載/讀取 MNIST 資料集...")
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    generator = Generator(latent_dim, num_classes)
    discriminator = Discriminator(num_classes)
    
    # 損失函數與優化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    
    print("開始訓練...")
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size_actual = imgs.size(0)
            
            # 真實資料的標籤為 1，假資料的標籤為 0
            valid = torch.ones(batch_size_actual, 1)
            fake = torch.zeros(batch_size_actual, 1)
            
            # --- 訓練生成器 ---
            optimizer_G.zero_grad()
            
            # 產生隨機雜訊與隨機標籤
            z = torch.randn(batch_size_actual, latent_dim)
            gen_labels = torch.randint(0, num_classes, (batch_size_actual,))
            
            # 產生假圖片
            gen_imgs = generator(z, gen_labels)
            
            # 生成器的目標是讓判別器把假圖片當成真的
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # --- 訓練判別器 ---
            optimizer_D.zero_grad()
            
            # 判別真實圖片
            real_pred = discriminator(imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)
            
            # 判別假圖片
            fake_pred = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            # 判別器總損失
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            if i % 300 == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
                
    # 儲存模型權重
    torch.save(generator.state_dict(), "models/generator.pth")
    print("訓練完成！模型權重已儲存至 models/generator.pth")

if __name__ == "__main__":
    train()
