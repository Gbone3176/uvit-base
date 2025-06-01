import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from open_clip import create_model_and_transforms, get_tokenizer
import os
os.CUDA_VISIBLE_DEVICES = '2'
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
    def __init__(self, version="StanfordAIMI/RadBERT", device="cuda", max_length=256):
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

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device) # (batch_size, max_length)
        outputs = self.bert_model(input_ids=tokens) # (batch_size, max_length, hidden_size)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):  
        return self(text) # 自动调用forward方法
    

# 以下是BERT系列的model_name
# michiyasunaga/BioLinkBERT-base 256
# michiyasunaga/BioLinkBERT-large 256 hidden_size 1024, 似乎不太可行
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# StanfordAIMI/RadBERT

# 测试代码
def test_frozen_clip_embedder():
    # 初始化编码器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BertEmbedder(device=device)
    embedder.to(device)

    # 测试文本
    text = [
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beautiful landscape"
    ]

    # 编码文本
    encoded_features = embedder.encode(text)

    # 检查输出
    print(f"Device: {device}")
    print(f"Encoded features shape: {encoded_features.shape}")  # 应为 (batch_size, max_length, hidden_size)
    print(f"Encoded features (first 5 tokens of first sentence): {encoded_features[0, :5, :5]}")

    # 检查是否冻结了模型参数
    for name, param in embedder.transformer.named_parameters():
        if param.requires_grad:
            print(f"Warning: Parameter {name} is not frozen!")
        else:
            print(f"Parameter {name} is frozen.")

    # 检查是否在正确的设备上
    assert encoded_features.device == torch.device(device), f"Features are not on {device}!"

    print("All tests passed!")


# 运行测试
if __name__ == "__main__":
    test_frozen_clip_embedder()