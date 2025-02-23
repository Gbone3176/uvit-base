import numpy as np
import torch
from torch import nn, tensor
from torch.autograd import Function, Variable
import torch.nn.functional as F
from math import log

class SPB(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, temp=0.9):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed  # 对于一个item的句子prototype
        self.decay = decay
        self.eps = eps
        self.embed = nn.Embedding(n_embed, dim)  # 通过embedding层构建prototype [num_prot, prot_dim]
        self.temp = temp
        self.curr_temp = temp

    def set_temp(self, epoch, max_epoch, strategy="fixed"):
        if strategy == "fixed":
            self.curr_temp = self.temp 
        elif strategy == "linear":
            self.curr_temp = self.temp - 0.9 * self.temp * epoch / max_epoch
        elif strategy == "exp":
            self.curr_temp = self.temp * (0.1 ** (epoch / max_epoch))
        
    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = flatten @ self.embed.weight.T
        self.gt_dist = dist
        soft_one_hot = F.gumbel_softmax(dist, tau=self.curr_temp, dim=1, hard=False)
        output = soft_one_hot @ self.embed.weight   # 加权每个prot，得到每行所对应的prot
        embed_ind = soft_one_hot.argmax(1)          # 选出每行对应的最有可能的那个prototype的index
        recon_loss = (output - flatten).abs().mean()
        loss = recon_loss
        self.dist = dist
        return output, loss, embed_ind

    @ torch.no_grad()
    def query(self, input):
        flatten = input.reshape(-1, self.dim)
        logits = flatten @ self.embed.weight.T  # embed就是prototype bank
        soft_one_hot = F.gumbel_softmax(logits, tau=self.curr_temp, dim=1, hard=False)
        output = soft_one_hot @ self.embed.weight
        #recon_loss = (output - flatten).abs().mean()
        embed_ind = soft_one_hot.argmax(1)
        return output, embed_ind


    def cal_loss(self, input):
        # calculate kl divergence loss between the qurey distribution and the ground truth distribution
        flatten = input.reshape(-1, self.dim)
        dist = flatten @ self.embed.weight.T
        log_gt_dist = F.log_softmax(self.gt_dist, dim=-1)
        log_dist = F.log_softmax(dist, dim=-1)
        kl_div = F.kl_div(log_dist, log_gt_dist, reduction='batchmean', log_target=True)
        return kl_div
    
    @ torch.no_grad()
    def save_weights(self, path="spb_weights.pth"):    
        torch.save(self.state_dict(), "/storage/PRIOR/codes/prior/modules/spb_weights/" + path)


if __name__=="__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Ensure that SPB class is defined before this testing code block,
    # Otherwise, you should import the SPB class if it's in another module.

    # Settings for the test
    dim = 768  # dimensionality of the input
    n_embed = 512  # number of prototypes
    batch_size = 2  # number of samples in a batch
    max_epoch = 100  # just for the temperature decay strategy

    # Initialize the SPB model
    model = SPB(dim, n_embed)
    torch.manual_seed(42)

    # Create some dummy input data
    # Simulate a batch of input vectors (batch_size, dim)
    input_data = torch.randn(batch_size, dim)
    print("Input shape:", input_data)
    # Set the temperature (you can choose the strategy)
    model.set_temp(epoch=10, max_epoch=max_epoch, strategy='fixed')

    # Run the forward pass
    output, loss, embed_ind = model(input_data)

    # Print outputs
    print("Output shape:", output.shape)
    print("Loss:", loss.item())
    print("Embedding Indices:", embed_ind)

    # Test the query method
    output_query, embed_ind_query = model.query(input_data)

    # Print query outputs
    print("Query Output:", output_query)
    print("Query Embedding Indices:", embed_ind_query)

    save_path = '/storage/PRIOR/codes/prior/modules/spb_weights/spb_weights_50.pth'
    model.save_weights(save_path)
    print(f'SPB module weights saved successfully')

    model2 = SPB(dim, n_embed)
    model2.load_state_dict(torch.load(save_path))
    print(f'SPB module weights loaded successfully')

    output_query2, embed_ind_query2 = model2.query(input_data)
    # Print query outputs

    print("Query Output:", output_query2)
    print("Query Embedding Indices:", embed_ind_query2)
    print(f"model = model2? {model == model2}")