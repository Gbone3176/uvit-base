import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from open_clip import create_model_and_transforms, get_tokenizer
import os
import re

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModel,
    

    CLIPPreTrainedModel,
    CLIPTextModel, 
    CLIPTextConfig,
    CLIPTokenizerFast, 
    PreTrainedModel, 
    CLIPConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import sys
# 将官方 timm 模块的路径添加到 sys.path 的最前面,避免导入libs下的timm
sys.path.insert(0, '/opt/conda/envs/uvit/lib/python3.10/site-packages/timm')

os.CUDA_VISIBLE_DEVICES = '2'


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device) # (batch_size, max_length)
        outputs = self.transformer(input_ids=tokens) # (batch_size, max_length, hidden_size)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):  
        return self(text) # 自动调用forward方法

class BioMedClipEmbedder(AbstractEncoder):
    """Uses the BiomedCLIP transformer encoder for text (from Hugging Face Hub)"""
    def __init__(self, version="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", device="cuda", max_length=256):
        super().__init__()
        # Load the model and tokenizer from Hugging Face Hub
        self.model, _, preprocess = create_model_and_transforms(version)
        self.encoder = self.model.text
        self.encoder.output_tokens = True

        self.tokenizer = get_tokenizer(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        """Freeze the model parameters to disable training."""
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def token(self, texts):
        """Tokenize the input text using the tokenizer."""
        inner_tokenizer = self.tokenizer.tokenizer
        tokens = []
        for text in texts:
            token_embeddings = inner_tokenizer(text, max_length=self.max_length)
            token = inner_tokenizer.convert_ids_to_tokens(token_embeddings["input_ids"])
            tokens.append(token)
        return tokens
    
    def forward(self, text):
        # Tokenize the input text
        token_embeddings = self.tokenizer(text, context_length=self.max_length).to(self.device)
        outputs = self.encoder(token_embeddings) # (batch_size, max_length, hidden_size)
        
        # Get the hidden states from the transformer
        z = outputs[1] # 取出hidden_size
        return z

    def encode(self, text):
        return self(text)

class PubMedClipEmbedder(AbstractEncoder):
    """Uses the PubMedCLIP transformer encoder for text embeddings"""
    def __init__(self, version="flaviagiammarino/pubmed-clip-vit-base-patch32", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device) # (batch_size, max_length)
        outputs = self.transformer(input_ids=tokens) # (batch_size, max_length, hidden_size)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):  
        return self(text) # 自动调用forward方法

class BertEmbedder(AbstractEncoder):
    """Uses the BERT transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="StanfordAIMI/RadBERT", device="cuda", max_length=256, mask=False):
        super().__init__()
        
        print("\n")
        print("**TextEmbedder**:", version)
        print("\n")

        # Load the model and tokenizer from Hugging Face Hub
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.bert_model = AutoModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.mask = mask

        self.freeze()

    def freeze(self):
        self.bert_model = self.bert_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device) # (batch_size, max_length)
        attn_mask = batch_encoding["attention_mask"].to(self.device)
        outputs = self.bert_model(input_ids=tokens, attention_mask=attn_mask) # (batch_size, max_length, hidden_size)

        z = outputs.last_hidden_state

        return z, attn_mask if self.mask else z

    def encode(self, text):  
        return self(text) # 自动调用forward方法
    
class BertEmbedder4Vis(AbstractEncoder):
    """Uses the BERT transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="michiyasunaga/BioLinkBERT-base", device="cuda", max_length=256):
        super().__init__()
        
        print("\n")
        print("**TextEmbedder**:", version)
        print("\n")

        # Load the model and tokenizer from Hugging Face Hub
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.bert_model = AutoModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length

        self.freeze()

    def freeze(self):
        self.bert_model = self.bert_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: str, word: str):
        # 1) 分词并带回字符偏移
        batch = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True
        )

        input_ids  = batch["input_ids"].to(self.device)            # [1, L]
        attn_mask  = batch["attention_mask"][0].tolist()           # [L]
        spec_mask  = batch["special_tokens_mask"][0].tolist()      # [L]
        offsets    = batch["offset_mapping"][0].tolist()           # [(start, end)] * L

        # 2) 在原文中找到目标词的所有出现区间（大小写不敏感，精确子串匹配）
        #    如果你想“整词匹配”，把下一行改为：rf"\b{re.escape(word)}\b"
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(word), text, flags=re.I)]

        # 3) 构造 token 级 mask：与任一目标区间有交集则置 1
        L = input_ids.size(1)
        token_mask = torch.zeros(L, dtype=torch.long, device=self.device)  # 0/1 掩码

        if spans:
            for i, ((s, e), sp, am) in enumerate(zip(offsets, spec_mask, attn_mask)):
                if am == 0:      # 到 PAD 之后提前结束
                    break
                if sp == 1:      # 跳过 [CLS]/[SEP] 等特殊 token
                    continue
                if s is None or e is None or e <= s:
                    continue
                # 与任一目标区间相交即可（区间相交条件：不满足互相分离）
                for (ws, we) in spans:
                    if not (e <= ws or we <= s):
                        token_mask[i] = 1
                        break

        # 4) 过 BERT，得到 last_hidden_state
        outputs = self.bert_model(input_ids=input_ids)
        z = outputs.last_hidden_state  # [1, L, H]

        return z, token_mask


    def encode(self, text, word):  
        return self(text, word) # 自动调用forward方法

# 以下是BERT系列的model_name
# michiyasunaga/BioLinkBERT-base 256
# michiyasunaga/BioLinkBERT-large 256 hidden_size 1024
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# StanfordAIMI/RadBERT

# 测试代码
def test_frozen_clip_embedder():
    # 初始化编码器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BertEmbedder4Vis(device=device)
    embedder.to(device)

    # 测试文本
    text1 = "chest x-ray showing pneumonia in the left lung"
    text2 = "chest x-ray showing pneumonia in the right lung"

    word = 'x-ray'
    # 编码文本
    encoded_features1, token_mask1 = embedder.encode(text1, word)
    encoded_features2, token_mask2 = embedder.encode(text2, word)

    cossim = torch.nn.functional.cosine_similarity(encoded_features1.squeeze(0)[1:8,:], encoded_features2.squeeze(0)[1:8,:], dim=-1).mean()
    # cossim = torch.nn.functional.cosine_similarity(encoded_features1.squeeze(0), encoded_features2.squeeze(0), dim=-1).mean()
    print(f"Cosine similarity between '{text1}' and '{text2}': {cossim.item()}")

    # # 检查输出
    # print(f"Device: {device}")
    # print(f"Encoded features shape: {encoded_features.shape}")  # 应为 (batch_size, max_length, hidden_size)
    # print(f"Encoded features (first 5 tokens of first sentence): {encoded_features[0, :5, :5]}")

    # # 检查是否冻结了模型参数
    # for name, param in embedder.transformer.named_parameters():
    #     if param.requires_grad:
    #         print(f"Warning: Parameter {name} is not frozen!")
    #     else:
    #         print(f"Parameter {name} is frozen.")

    # # 检查是否在正确的设备上
    # assert encoded_features.device == torch.device(device), f"Features are not on {device}!"

    # print("All tests passed!")


# 运行测试
if __name__ == "__main__":
    test_frozen_clip_embedder()