import torch
import torch.nn as nn
import torch.nn.functional as F

class attention(nn.Module): 
    def __init__(self, embed_size, head): 
        super().__init__()
        self.multi_head = head 
        self.embed_size = embed_size 
        self.head_dim = embed_size // head 
        
        self.W_Q = nn.Linear(embed_size, embed_size)
        self.W_K = nn.Linear(embed_size, embed_size) 
        self.W_V = nn.Linear(embed_size, embed_size) 
        self.W_O = nn.Linear(embed_size, embed_size)
    def forward(self, x): 
        # (B, L, E) -> <B, N, L, E/N> 
        Q = self.W_Q(x).view(x.shape[0], self.multi_head, x.shape[1], self.head_dim)
        K = self.W_K(x).view(x.shape[0], self.multi_head, x.shape[1], self.head_dim)
        V = self.W_V(x).view(x.shape[0], self.multi_head, x.shape[1], self.head_dim)
        
        scores = F.softmax(Q @ K.transpose(-2, -1) / self.head_dim ** 0.5, dim = -1) 
        out = scores @ V
        # <B, N, L, E/N> -> (B, L, E)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], -1, self.embed_size)
        return self.W_O(out) 
    
class masked_attention(attention): 
    def forward(self, x):  
        B, L, E = x.shape
        Q = self.W_Q(x).view(B, self.multi_head, L, self.head_dim) 
        K = self.W_K(x).view(B, self.multi_head, L, self.head_dim)
        V = self.W_V(x).view(B, self.multi_head, L, self.head_dim) 
        
        Mask = torch.triu(torch.ones(L, L), diagonal = 1)
        Mask = Mask.masked_fill(Mask == 1, float('-inf'))
        Mask = Mask.unsqueeze(0).unsqueeze(1)
        QK_mul = Q @ K.transpose(-2, -1) / self.head_dim ** 0.5 
        QK_mul = QK_mul + Mask 
        scores = F.softmax(QK_mul, dim = -1) 
        out =  scores @ V 
        return self.W_O(out.view(B, L, E))
class cross_attention(attention):
    def forward(self, x1, x2):
        B, L1, E = x1.shape
        L2 = x2.shape[1]
        
        Q = self.W_Q(x1).view(B, self.multi_head, L1, self.head_dim)
        K = self.W_K(x2).view(B, self.multi_head, L2, self.head_dim)
        V = self.W_V(x2).view(B, self.multi_head, L2, self.head_dim)
        
        scores = F.softmax(Q @ K.transpose(-2, -1) / self.head_dim ** 0.5, dim = -1) 
        
        out = scores @ V
        out = out.transpose(1, 2).view(B, L1, E)
        return self.W_O(out) 
    
class transformer_layer(nn.Module): 
    def __init__(self, self, embed_size, head, forward_dim, causal = False):
        super().__init__() 
        self.layer_norm = nn.LayerNorm(embed_size) 
        self.dropout = nn.Dropout(0.5)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, embed_size) 
        )
        if causal:
            self.attention = masked_attention(embed_size, head)
        else:
            self.attention = attention(embed_size, head) 
    def forward(self, x):        
        x = x + self.dropout(self.attention(self.layer_norm(x)))
        x = x + self.dropout(self.feed_forward(self.layer_norm(x)))
        return x

class Transformer(nn.Module): 
    def __init__(self, n, L, embed_size, head, forward_dim, causal = False): 
        self.embedding = nn.Embedding(L, embed_size)
        self.encoder_attention = nn.ModuleList([
        transformer_layer(embed_size, head, forward_dim, causal) for i in range(n)
        ])
        
    def position_encoding(self, x):     
        B, L, E = x.shape
        dim_index = torch.arange(0, E, 2).unsqueeze(0)
        # exp^(-dim / E * log(10000) = exp^(log(10000)^ (-dim / E))) = 1 / 10000 ^ (dim / E)
        div_term = torch.exp(-dim_index / E * torch.log(10000))
        pos = torch.arange(0, L).unsqueeze(1) 
        pos_encode = torch.zeros(L, E, device=device)
        pos_encode[:, 0::2] = torch.sin(pos * div_term)
        pos_encode[:, 1::2] = torch.cos(pos * div_term)
        return x + pos_encode.unsqueeze(0)
    def forward(self, x): 
        x = self.embedding(x)
        x = self.position_encoding(x) 
        x = self.encoder_attention(x)
        return x
        