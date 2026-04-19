import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler  # 💥 核武器：混合精度加速
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
from dataset import Flickr8kDataset
from model import VSEModel

matplotlib.use('Agg')
# 屏蔽烦人的系统警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==========================================
# ⚙️ 核心超参数配置区 (专为 RTX 3090/4090 定制)
# ==========================================
JSON_PATH = '/root/autodl-tmp/my_vse_project/flickr8k_aim3/flickr8k_aim3/dataset_flickr8k.json'
IMG_DIR = '/root/autodl-tmp/my_vse_project/flickr8k_aim3/flickr8k_aim3/images'
BATCH_SIZE = 256  # 24GB显存随便开 64 或 128！对比学习的命脉！
EMBED_DIM = 512  # 联合语义空间维度 (越大越聪明)
LEARNING_RATE = 2e-5  # 双流模型微调，学习率必须极小！(千万别用0.01)
NUM_EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def info_nce_loss(img_emb, txt_emb, logit_scale):
    # 直接用学到的温度来缩放！
    logits = logit_scale * torch.matmul(img_emb, txt_emb.T)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0

def main():

    print(f"🔥 炼丹炉点火！当前使用引擎: {device.type.upper()}")

    # 1. 准备滤镜与发牌机
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Flickr8kDataset(JSON_PATH, IMG_DIR, split='train', transform=train_transform)

    # 💥 drop_last=True 极其重要！防止最后一个 Batch 不满 64 导致 Loss 矩阵对不齐报错！
    # pin_memory=True 让 CPU 和 GPU 内存直连，数据传输光速飙升！
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)

    # 2. 召唤同声传译与双流大脑
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = VSEModel(embed_dim=EMBED_DIM).to(device)

    # 3. 聘请教练与变速箱
    optimizer = optim.AdamW([
        {'params': model.cnn.parameters(), 'lr': 1e-5},
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.img_proj.parameters(), 'lr': 5e-4},
        {'params': model.txt_proj.parameters(), 'lr': 5e-4},
        {'params': [model.logit_scale], 'lr': 5e-4}
    ], weight_decay=1e-4)
    scaler = GradScaler()  # AMP 混合精度发电机

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    # 4. 开启地狱循环
    print("\n⚔️ 跨模态相亲大会正式开始...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, captions) in enumerate(train_loader):
            # 搬运数据
            images = images.to(device)
            text_inputs = tokenizer(captions, padding=True, truncation=True,
                                    max_length=32, return_tensors="pt").to(device)

            optimizer.zero_grad()

            # 💥 启动自动混合精度 (AMP) 结界！显存占用减半，速度翻倍！
            with autocast():
                img_emb, txt_emb,TEMPERATURE = model(images, text_inputs["input_ids"], text_inputs["attention_mask"])
                loss = info_nce_loss(img_emb, txt_emb, logit_scale=TEMPERATURE)

            # AMP 专属的反向传播与拧旋钮
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # 每 10 个 Batch 汇报一次战况
            if batch_idx % 10 == 0:
                print(
                    f"Epoch[{epoch + 1}/{NUM_EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Epoch 总结
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"🚀 Epoch {epoch + 1} 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {epoch_time:.0f}秒\n")

        scheduler.step()

    print("🎉 炼丹完成！你的大模型已经成功掌握了跨越图文物种的加密语言！")
    torch.save(model.state_dict(), f'vse_multimodal_model_size:{BATCH_SIZE}.pth')

if __name__ == '__main__':
    main()