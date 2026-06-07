import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from cgan_model import Generator

def main():
    # 建立用來存放靜態圖片的資料夾
    os.makedirs("assets/generated", exist_ok=True)
    os.makedirs("assets/real", exist_ok=True)
    
    latent_dim = 100
    generator = Generator(latent_dim)
    model_path = "models/generator.pth"
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        generator.eval()
        print("已成功載入訓練好的 GAN 模型！")
    else:
        print("警告：找不到訓練好的模型 (models/generator.pth)。將產生初始雜訊圖片。")
        generator.eval()

    print("開始產生生成器 (GAN) 的靜態圖片...")
    num_samples = 10 # 每個數字產生 10 張供網頁隨機抽取
    
    with torch.no_grad():
        for digit in range(10):
            for i in range(num_samples):
                # 產生隨機雜訊與指定的標籤
                z = torch.randn(1, latent_dim)
                labels = torch.tensor([digit], dtype=torch.long)
                
                # 生成圖片
                gen_img = generator(z, labels)
                
                # 將範圍從 [-1, 1] 轉換回 [0, 1]
                gen_img = (gen_img + 1) / 2.0
                
                # 存檔
                pil_img = to_pil_image(gen_img.squeeze(0))
                pil_img.save(f"assets/generated/digit_{digit}_gen_{i}.png")

    print("開始擷取真實資料集 (MNIST) 的圖片...")
    # 下載或載入 MNIST
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    
    # 用字典來計算每個數字已經抓了幾張
    counts = {i: 0 for i in range(10)}
    
    for img, label in dataset:
        if counts[label] < num_samples:
            # 存檔
            pil_img = to_pil_image(img)
            pil_img.save(f"assets/real/digit_{label}_real_{counts[label]}.png")
            counts[label] += 1
            
        # 如果每個數字都抓滿了就提早結束
        if all(count == num_samples for count in counts.values()):
            break

    print("所有靜態圖片已產生完畢！儲存於 assets/ 資料夾中。")

if __name__ == "__main__":
    main()
