import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import RESULTS_DIR, LEARNING_RATE, EPOCHS, BATCH_SIZE


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for user_idx, artist_idx, tag_features, labels in loader:
        user_idx    = user_idx.to(device)
        artist_idx  = artist_idx.to(device)
        tag_features = tag_features.to(device)
        labels      = labels.to(device)

        optimizer.zero_grad()
        preds = model(user_idx, artist_idx, tag_features)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, dataset, epochs=EPOCHS, lr=LEARNING_RATE,
          batch_size=BATCH_SIZE, device='cpu'):
    """
    Entrena el modelo y guarda el mejor checkpoint en results/best_model.pt.
    Devuelve el historial de pérdida por época.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    history   = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        history.append(loss)
        print(f'Epoch {epoch:02d}/{epochs}  Loss: {loss:.4f}')

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pt')

    print(f'\nMejor loss: {best_loss:.4f}  →  guardado en results/best_model.pt')
    return history
