import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image

class Flickr8kDataset(Dataset):
    def __init__(self, json_path, img_dir, split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data_pairs =[]
        self.split = split

        print(f"📂 正在加载 {split} 阶段的 JSON 数据...")
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for item in dataset['images']:
            if item['split'] == split:
                filename = item['filename']
                # 💥 核心改变：不压平了！把 5 句话打包成一个列表存起来！
                captions = [sentence['raw'] for sentence in item['sentences']]
                self.data_pairs.append((filename, captions))

        print(f"✅ {split} 数据集加载完毕！共提取了 {len(self.data_pairs)} 张独立图片。")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        filename, captions_list = self.data_pairs[idx]
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 💥 核心修复：考场严禁掷骰子！
        # 假设你之前在 __init__ 里存了一个 self.split = split
        if self.split == 'train':
            chosen_caption = random.choice(captions_list) # 训练时随机抽，增强泛化
        else:
            chosen_caption = captions_list[0] # 考试时，永远只取第一句话，保证成绩绝对稳定！

        return image, chosen_caption
