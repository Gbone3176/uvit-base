import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from open_clip import create_model_and_transforms, get_tokenizer
import os
from attention_visualizer import display_attention


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

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        attention_mask = batch_encoding["attention_mask"].to(self.device)
        tokens = batch_encoding["input_ids"].to(self.device)  # (batch_size, max_length)
        outputs = self.bert_model(input_ids=tokens, attention_mask=attention_mask)  # (batch_size, max_length, hidden_size)
        
        z = outputs.last_hidden_state
        
        # 返回原始tokens和特征，以及attention_mask，让调用者知道哪些是padding
        return tokens, z, attention_mask

    def encode(self, text):  
        return self(text) # 自动调用forward方法

os.CUDA_VISIBLE_DEVICES = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"

# michiyasunaga/BioLinkBERT-base 
# michiyasunaga/BioLinkBERT-large (hidden_size 1024, 似乎不太可行)
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# StanfordAIMI/RadBERT

def visualize_text_attention(text, version):
    clip = BertEmbedder(version)
    clip.eval()
    clip.to(device)
    
    with torch.no_grad():
        tokens, features, attention_mask = clip.encode(text)
    
    # 只使用第一个样本的非padding部分
    valid_length = attention_mask[0].sum().item()
    valid_tokens = tokens[0, :valid_length]
    valid_features = features[0, :valid_length]
    
    tokens_list = valid_tokens.cpu().numpy().tolist()
    decoded_tokens = clip.tokenizer.convert_ids_to_tokens(tokens_list)
    
    # 计算token重要性
    raw_importance = torch.norm(valid_features, dim=1)
    token_importance = (raw_importance - raw_importance.min()) / (raw_importance.max() - raw_importance.min())
    token_importance = token_importance.cpu().numpy().tolist()
    
    display_attention(decoded_tokens, token_importance)

# 使用示例
search_query = "A red flower is under the blue sky and there is a bee on the flower"
visualize_text_attention(search_query, "StanfordAIMI/RadBERT")
