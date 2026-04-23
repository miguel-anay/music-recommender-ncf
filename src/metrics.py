import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from config import TOP_K, N_CANDIDATES, RANDOM_SEED


def _rank_positive(model, user_idx, pos_artist_idx, tag_matrix,
                   n_candidates=N_CANDIDATES, device='cpu'):
    """
    Rankea el artista positivo contra (n_candidates - 1) negativos aleatorios.
    Devuelve la posición (0-indexed) del positivo en el ranking.
    """
    n_artists = len(tag_matrix)
    all_idx   = np.arange(n_artists)
    neg_pool  = all_idx[all_idx != pos_artist_idx]
    neg_sample = np.random.choice(neg_pool, size=n_candidates - 1, replace=False)
    candidates = np.concatenate([[pos_artist_idx], neg_sample])

    model.eval()
    with torch.no_grad():
        u = torch.tensor([user_idx] * len(candidates), dtype=torch.long).to(device)
        a = torch.tensor(candidates,                   dtype=torch.long).to(device)
        t = torch.tensor(tag_matrix[candidates],       dtype=torch.float32).to(device)
        scores = model(u, a, t).cpu().numpy()

    ranked    = np.argsort(scores)[::-1]
    pos_rank  = int(np.where(np.array(candidates)[ranked] == pos_artist_idx)[0][0])
    return pos_rank


def hit_rate_at_k(model, test_pairs, tag_matrix, k=TOP_K,
                  n_candidates=N_CANDIDATES, device='cpu',
                  random_state=RANDOM_SEED):
    """Fracción de usuarios cuyo artista positivo aparece en el top-K."""
    np.random.seed(random_state)
    hits = sum(
        1 for u, a in test_pairs
        if _rank_positive(model, u, a, tag_matrix, n_candidates, device) < k
    )
    return hits / len(test_pairs) if test_pairs else 0.0


def ndcg_at_k(model, test_pairs, tag_matrix, k=TOP_K,
              n_candidates=N_CANDIDATES, device='cpu',
              random_state=RANDOM_SEED):
    """NDCG@K usando descuento logarítmico sobre la posición del positivo."""
    np.random.seed(random_state)
    scores = []
    for u, a in test_pairs:
        rank = _rank_positive(model, u, a, tag_matrix, n_candidates, device)
        scores.append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_svd_baseline(user_item_log, test_pairs_raw,
                          user_to_idx, artist_to_idx,
                          k=TOP_K, n_candidates=N_CANDIDATES,
                          random_state=RANDOM_SEED):
    """
    Evalúa TruncatedSVD como baseline con el mismo protocolo que NCF.
    Devuelve dict con 'hit_rate' y 'ndcg'.
    """
    np.random.seed(random_state)

    svd            = TruncatedSVD(n_components=50, random_state=42)
    user_factors   = svd.fit_transform(user_item_log)
    artist_factors = svd.components_.T
    n_artists      = artist_factors.shape[0]

    hits, ndcg_scores = 0, []

    for user_id, artist_id in test_pairs_raw:
        if user_id not in user_to_idx or artist_id not in artist_to_idx:
            continue

        u_idx = user_to_idx[user_id]
        a_idx = artist_to_idx[artist_id]

        neg_pool   = np.arange(n_artists)
        neg_pool   = neg_pool[neg_pool != a_idx]
        neg_sample = np.random.choice(neg_pool, size=n_candidates - 1, replace=False)
        candidates = np.concatenate([[a_idx], neg_sample])

        scores  = user_factors[u_idx] @ artist_factors[candidates].T
        ranked  = np.argsort(scores)[::-1]
        rank    = int(np.where(np.array(candidates)[ranked] == a_idx)[0][0])

        if rank < k:
            hits += 1
            ndcg_scores.append(1.0 / np.log2(rank + 2))
        else:
            ndcg_scores.append(0.0)

    total = len(ndcg_scores)
    return {
        'hit_rate': hits / total if total > 0 else 0.0,
        'ndcg':     float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }
