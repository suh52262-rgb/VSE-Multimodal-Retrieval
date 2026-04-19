import torch
import torch.nn as nn
from torchvision import  models
from transformers import AutoModel

class VSEModel(nn.Module):
    def __init__(self, embed_dim=512):
        super(VSEModel, self).__init__()

        # 1. 视觉大爹：升级为更深邃的 ResNet50
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # 极其优雅的砍头手术
        self.img_proj = nn.Linear(2048, embed_dim)  # ResNet50输出是2048维

        # 2. 语言大爹：极其轻量的 DistilBERT
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.txt_proj = nn.Linear(768, embed_dim)

        # 💥 OpenAI 同款黑科技：可学习的温度参数！
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, images, input_ids, attention_mask):
        # 图像流提纯
        img_feats = self.img_proj(self.cnn(images))

        # 文本流提纯 (取 [CLS] 班长作为整句话的灵魂)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = self.txt_proj(bert_output.last_hidden_state[:, 0, :])

        # 💥 降维打击：强行化为单位向量，只拼方向(语义)，不拼大小！
        img_feats = nn.functional.normalize(img_feats, p=2, dim=-1)
        txt_feats = nn.functional.normalize(txt_feats, p=2, dim=-1)

        return img_feats, txt_feats,self.logit_scale.exp()
