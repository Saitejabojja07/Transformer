import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Converts token IDs into dense embedding vectors of dimension `d_model`.

    Parameters
    ----------
    d_model : int
        The dimensionality of each token embedding vector.  
        Example: if d_model = 512, every token becomes a 512-dimensional vector.

    vocab_size : int
        The total number of tokens in the vocabulary.
        This determines how many unique embeddings are stored.

    Attributes
    ----------
    embeddings : nn.Embedding
        Lookup table of shape (vocab_size, d_model) that maps each token ID
        to a learnable vector. These vectors are updated during training.
    
    Notes
    -----
    • The forward method returns embeddings scaled by √d_model,
      as done in the original Transformer paper to stabilize training.

    • Padding tokens (ID = 0) are assigned a fixed zero-vector embedding
      that does not update during training.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model

        # Each token ID → a vector of size d_model
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes token embeddings.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of token IDs of shape:
                (batch_size, seq_len)

            • batch_size = number of sequences processed at once  
            • seq_len = length of each token sequence  
            Exampl:
                [[5,  9, 2, 0],
                 [3, 11, 4, 7]]

        Returns
        -------
        torch.Tensor
            Embedded token representations of shape:
                (batch_size, seq_len, d_model)

            For example, if:
                batch_size = 32
                seq_len = 20
                d_model = 512

            Output shape = (32, 20, 512)

            Each token ID is replaced by its learned embedding vector.
            Embeddings are scaled by sqrt(d_model) as recommended in
            the Transformer architecture.
        """
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to token embeddings as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Dimensionality of the embedding vectors. Must match the input embedding size.

    seq : int
        Maximum sequence length supported by the model.
        Positional encodings for all positions 0 ... seq-1 are precomputed.

    dropout : float
        Dropout probability applied after adding positional encodings.

    Attributes
    ----------
    pe : torch.Tensor
        Positional encoding matrix of shape (1, seq, d_model), registered as a buffer.
        It is not a learnable parameter and does not receive gradients.

    Notes
    -----
    • Uses sine for even indices and cosine for odd indices.
    • Positional encodings provide the model with information about token order.
    • The buffer ensures `pe` is saved with the model but excluded from gradient updates.
    """

    def __init__(self, d_model: int, seq: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        # Create matrix (seq, d_model)
        pe = torch.zeros(seq, d_model)

        # position: (seq, 1)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension → (1, seq, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (non-trainable)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape:
                (batch_size, seq_len, d_model)

            Typically the output of the token embedding layer.

        Returns
        -------
        torch.Tensor
            Same shape as input:
                (batch_size, seq_len, d_model)

            Positional encodings are added element-wise to x.
        """
        # Add positional encoding for the first seq_len positions
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

        
        