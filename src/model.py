import torch
import torch.nn as nn
from config import EMBED_DIM, TAG_FEATURES


class NCFHybrid(nn.Module):
    """
    Neural Collaborative Filtering híbrido.
    Combina embeddings de usuario, artista y features de tags (TF-IDF)
    a través de un MLP para predecir feedback implícito.
    """

    def __init__(self, n_users, n_artists, embed_dim=EMBED_DIM, tag_dim=TAG_FEATURES):
        super().__init__()

        self.user_embedding   = nn.Embedding(n_users,   embed_dim)
        self.artist_embedding = nn.Embedding(n_artists, embed_dim)
        self.tag_projection   = nn.Linear(tag_dim, embed_dim, bias=False)

        # MLP: concat(user, artist, tag) → 192 → 128 → 64 → 1
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight,   std=0.01)
        nn.init.normal_(self.artist_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.tag_projection.weight)

    def forward(self, user_idx, artist_idx, tag_features):
        """
        Args:
            user_idx:    LongTensor (batch,)
            artist_idx:  LongTensor (batch,)
            tag_features: FloatTensor (batch, tag_dim)
        Returns:
            FloatTensor (batch,) con valores en [0, 1]
        """
        u   = self.user_embedding(user_idx)
        a   = self.artist_embedding(artist_idx)
        t   = torch.relu(self.tag_projection(tag_features))
        x   = torch.cat([u, a, t], dim=1)
        return self.mlp(x).squeeze(1)
