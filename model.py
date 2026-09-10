
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim=4, num_heads=2, ff_dim=8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape

        # Q, K, V
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        # Split into multiple heads
        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scaling
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Softmax
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        head_outputs = attention_weights @ V

        # Combine heads
        head_outputs = head_outputs.transpose(1, 2)

        combined_output = head_outputs.contiguous().view(
            batch_size,
            seq_len,
            embed_dim
        )

        # Output projection
        attention_output = self.output_projection(
            combined_output
        )

        # Residual + LayerNorm
        x = self.layer_norm1(
            x + attention_output
        )

        # Feed Forward Network
        ffn_output = self.ffn(x)

        # Residual + LayerNorm
        x = self.layer_norm2(
            x + ffn_output
        )

        return x


class TinyTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim=4,
        num_heads=2,
        ff_dim=8,
        num_blocks=2
    ):
        super().__init__()

        self.context_length = context_length

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                ff_dim
            )
            for _ in range(num_blocks)
        ])

        # Language model head
        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size
        )


    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeddings = self.token_embedding(
            input_ids
        )

        # Position embeddings
        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token + position information
        x = token_embeddings + position_embeddings

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Convert hidden states to vocabulary logits
        logits = self.lm_head(x)

        return logits
