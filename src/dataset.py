import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset
from config import DATA_DIR, MIN_INTERACTIONS, TAG_FEATURES, NEG_RATIO, TEST_SIZE, RANDOM_SEED


def load_and_filter(min_interactions=MIN_INTERACTIONS):
    """Carga user_artists.dat y filtra usuarios/artistas con pocas interacciones."""
    ua = pd.read_csv(DATA_DIR / 'user_artists.dat', sep='\t', encoding='latin-1')

    valid_users   = ua.groupby('userID').size()
    valid_artists = ua.groupby('artistID').size()

    ua = ua[
        ua['userID'].isin(valid_users[valid_users   >= min_interactions].index) &
        ua['artistID'].isin(valid_artists[valid_artists >= min_interactions].index)
    ].copy()

    if ua.empty:
        raise ValueError(f"No quedan interacciones con min_interactions={min_interactions}")

    return ua.reset_index(drop=True)


def binarize(df):
    """Aplica log1p y binariza por mediana → feedback implícito binario."""
    df = df.copy()
    df['weight_log'] = np.log1p(df['weight'])
    median = df['weight_log'].median()
    df['label'] = (df['weight_log'] >= median).astype(float)
    return df


def build_tag_features(artist_ids):
    """Construye matriz TF-IDF de tags por artista (n_artists × TAG_FEATURES)."""
    user_tagged = pd.read_csv(DATA_DIR / 'user_taggedartists.dat', sep='\t', encoding='latin-1')
    tags_df     = pd.read_csv(DATA_DIR / 'tags.dat',               sep='\t', encoding='latin-1')

    merged = user_tagged.merge(tags_df, on='tagID', how='left')
    artist_docs = (merged.groupby('artistID')['tagValue']
                         .apply(lambda x: ' '.join(x.astype(str)))
                         .reset_index()
                         .rename(columns={'tagValue': 'tag_doc'}))

    base = pd.DataFrame({'artistID': artist_ids})
    base = base.merge(artist_docs, on='artistID', how='left')
    base['tag_doc'] = base['tag_doc'].fillna('')

    vectorizer  = TfidfVectorizer(max_features=TAG_FEATURES)
    tag_matrix  = vectorizer.fit_transform(base['tag_doc']).toarray().astype(np.float32)

    return tag_matrix, vectorizer


def split_by_user(df, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Reserva test_size fracción de positivos de cada usuario para test."""
    train_parts, test_parts = [], []

    for _, group in df.groupby('userID'):
        pos = group[group['label'] == 1]
        neg = group[group['label'] == 0]

        if len(pos) < 2:
            train_parts.append(group)
            continue

        n_test  = max(1, int(len(pos) * test_size))
        test_p  = pos.sample(n=n_test, random_state=random_state)
        train_p = pos.drop(test_p.index)

        train_parts.append(pd.concat([train_p, neg]))
        test_parts.append(test_p)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df  = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame()
    return train_df, test_df


class LastFMDataset(Dataset):
    """Dataset PyTorch con negative sampling para entrenamiento NCF."""

    def __init__(self, interactions, tag_matrix, user_to_idx, artist_to_idx,
                 neg_ratio=NEG_RATIO, random_state=RANDOM_SEED):
        np.random.seed(random_state)
        self.tag_matrix   = tag_matrix
        self.user_to_idx  = user_to_idx
        self.artist_to_idx = artist_to_idx

        positives = interactions[interactions['label'] == 1][['userID', 'artistID']].values
        user_pos_set = {}
        for uid, aid in positives:
            user_pos_set.setdefault(uid, set()).add(aid)

        all_artist_ids = list(artist_to_idx.keys())
        pairs = [(u, a, 1.0) for u, a in positives]

        for uid, aid in positives:
            seen   = user_pos_set[uid]
            sampled = 0
            while sampled < neg_ratio:
                neg = np.random.choice(all_artist_ids)
                if neg not in seen:
                    pairs.append((uid, neg, 0.0))
                    sampled += 1

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid, aid, label = self.pairs[idx]
        u_idx = self.user_to_idx[uid]
        a_idx = self.artist_to_idx[aid]
        return (
            torch.tensor(u_idx,                             dtype=torch.long),
            torch.tensor(a_idx,                             dtype=torch.long),
            torch.tensor(self.tag_matrix[a_idx],            dtype=torch.float32),
            torch.tensor(label,                             dtype=torch.float32),
        )
