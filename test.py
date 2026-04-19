import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#选择镜像站以及忽略警告
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from dataset import Flickr8kDataset
from model import VSEModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 测试引擎启动: {device.type.upper()}")

def main():
    print("1. 正在唤醒大模型...")
    model = VSEModel(embed_dim=512).to(device)

    model.load_state_dict(torch.load('vse_multimodal_model_size:256.pth', map_location=device))

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("2. 正在准备测试试卷...")
    # 注意：测试集绝不加 RandomFlip 等数据增强
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    JSON_PATH = '/root/autodl-tmp/my_vse_project/flickr8k_aim3/flickr8k_aim3/dataset_flickr8k.json'
    IMG_DIR = '/root/autodl-tmp/my_vse_project/flickr8k_aim3/flickr8k_aim3/images'
    test_dataset = Flickr8kDataset(JSON_PATH, IMG_DIR, split='val', transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print("3. 机器正在疯狂阅读所有图片和文字，提取 512 维灵魂特征...")

    all_img_embs = []
    all_txt_embs = []

    with torch.no_grad():  # 考试绝对不准记账算梯度，极度省显存！
        for images, captions in test_loader:
            images = images.to(device)
            text_inputs = tokenizer(captions, padding=True, truncation=True,
                                    max_length=32, return_tensors="pt").to(device)

            # 前向传播，吐出 512 维的纯粹灵魂
            img_emb, txt_emb, temperate= model(images, text_inputs["input_ids"], text_inputs["attention_mask"])

            # 把算出来的灵魂特征搬回 CPU 内存并存起来
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())

    # 把所有小 Batch 拼接成一个超级大矩阵 (比如 [1000, 512])
    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_txt_embs = torch.cat(all_txt_embs, dim=0)
    print(f"✅ 提取完毕！图片矩阵: {all_img_embs.shape}, 文字矩阵: {all_txt_embs.shape}")

    print("\n4. 正在计算全局余弦相似度矩阵...")
    sim_matrix = torch.matmul(all_img_embs, all_txt_embs.T)

    def calculate_recall_at_k(similarity_matrix, k):
        """
        算分逻辑：对于每一行（每一张图），从高到低排个名。
        看看真正的“原配（对角线上的那个）”有没有挤进前 K 名！
        """
        N = similarity_matrix.shape[0]
        correct = 0
        for i in range(N):
            # 取出第 i 张图对所有文字的打分，挑出分数最高的 K 个索引
            top_k_indices = torch.topk(similarity_matrix[i], k).indices
            # 如果真正的原配 (索引 i) 在这 K 个人里面，算作猜对！
            if i in top_k_indices:
                correct += 1
        return (correct / N) * 100

    print("\n====== 🏆 终极成绩单 (Image-to-Text 图找文) ======")
    print(f"Recall@1  (第一眼就认出原配): {calculate_recall_at_k(sim_matrix, 1):.2f}%")
    print(f"Recall@5  (前五名里包含原配): {calculate_recall_at_k(sim_matrix, 5):.2f}%")
    print(f"Recall@10 (前十名里包含原配): {calculate_recall_at_k(sim_matrix, 10):.2f}%")

    print("\n====== 🏆 终极成绩单 (Text-to-Image 文找图) ======")
    # 矩阵转置一下，逻辑瞬间变成“文找图”！
    sim_matrix_t = sim_matrix.T
    print(f"Recall@1  (第一眼就认出原配): {calculate_recall_at_k(sim_matrix_t, 1):.2f}%")
    print(f"Recall@5  (前五名里包含原配): {calculate_recall_at_k(sim_matrix_t, 5):.2f}%")
    print(f"Recall@10 (前十名里包含原配): {calculate_recall_at_k(sim_matrix_t, 10):.2f}%")

if __name__ == '__main__':
    main()