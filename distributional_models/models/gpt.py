import torch
import torch.nn as nn
from torch.nn import functional
from .neural_network import NeuralNetwork


class GPT(NeuralNetwork):
    def __init__(self,
                 corpus,
                 block_size,
                 embedding_size,
                 num_heads,
                 attention_size,
                 hidden_size,
                 weight_init,
                 device):

        # you passed all these in
        # vocab_size, embed_size, block_size, attention_size, hidden_size, num_head, device

        super(GPT, self).__init__(corpus, device)
        self.model_type = 'gpt'
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.attention_size = attention_size
        self.hidden_size = hidden_size
        self.block_size = block_size

        self.define_network()
        self.set_device(device)

    def define_network(self):
        self.layer_dict['token_embeddings_table'] = nn.Embedding(self.vocab_size, self.embedding_size)
        self.layer_dict['position_embeddings_table'] = nn.Embedding(self.block_size, self.embedding_size)
        self.layer_dict['combined_input_module'] = CombinedInput(self.layer_dict['token_embeddings_table'],
                                                                 self.layer_dict['position_embeddings_table'])
        self.layer_dict['attention_weighted_values'] = MultiHeadAttention(self.num_heads, self.attention_size // self.num_heads,
                                                        self.embedding_size, self.block_size)
        self.layer_dict['hidden_layer'] = FeedForward(self.attention_size, self.hidden_size)
        self.layer_dict['output'] = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x_window):
        _, block_size = x_window.shape
        combined_input = self.layer_dict['combined_input_module'](x_window, block_size)
        attention_weighted_values, attention_weights = self.layer_dict['attention_weighted_values'](combined_input)
        h = self.layer_dict['hidden_layer'](attention_weighted_values)
        outputs = self.layer_dict['output'](h)  # (Batch*Time*vocab_size)
        # batch_size, T, C = outputs.shape
        # logits = outputs.view(batch_size * T, C)
        # print("outputs:", outputs.shape)
        # print("logits:", logits.shape)
        #lstm_out = lstm_out[:, -1, :]
        # targets = targets.view(B * T)
        #
        # loss = functional.cross_entropy(logits, targets)
        # o_prob = torch.nn.functional.softmax(logits, dim=1)
        return outputs


class CombinedInput(nn.Module):
    def __init__(self, token_embeddings_table, position_embeddings_table):
        super().__init__()
        self.token_embeddings_table = token_embeddings_table
        self.position_embeddings_table = position_embeddings_table

    def forward(self, idx, T):
        token_embed = self.token_embeddings_table(idx)
        position_embed = self.position_embeddings_table(torch.arange(T))
        combined_input = token_embed + position_embed
        return combined_input


class Head(nn.Module):
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_mask_attention_weights = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B ,T, C)
        q = self.query(x)  # (B ,T, C)
        v = self.value(x)  # (B ,T, C)
        # wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled attention, original in paper "attention is all you need"
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B ,T, C) @  (B, C, T) -> (B, T, T)
        self.pre_mask_attention_weights = wei.clone()
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = functional.softmax(wei, dim=1)  # (B, T, T)

        out = wei @ v  # (B, T, T) @ (B, T ,C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for i in range(num_heads)])

    def forward(self, x):
        attention_weights_list = []
        head_outputs = []
        for head in self.heads:
            head_output = head(x)
            head_outputs.append(head_output)

            # Assuming pre_mask_attention_weights are stored in each head
            attention_weights_list.append(head.pre_mask_attention_weights)

        # Concatenating the outputs from all heads
        combined_output = torch.cat(head_outputs, dim=-1)

        # Average the attention weights across all heads
        combined_attention_weights = torch.cat(attention_weights_list, dim=-1)
        return combined_output, combined_attention_weights


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_size, hidden_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)
