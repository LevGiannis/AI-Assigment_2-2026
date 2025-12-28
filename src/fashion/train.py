import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.fashion.data import get_fashionmnist_loaders, set_seed
from src.fashion.model import FashionCNN

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    set_seed(config.get('seed', 42))
    device = get_device()
    print(f"Using device: {device}")
    train_loader, dev_loader, test_loader, n_train, n_dev, n_test = get_fashionmnist_loaders(
        data_dir='data', batch_size=config['batch_size'], dev_ratio=0.1, quick=args.quick, seed=config.get('seed', 42))
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    best_dev = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'dev_loss': [], 'best_epoch': 0}
    num_epochs = 2 if args.quick else config['epochs']
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                dev_losses.append(loss.item())
        avg_train = np.mean(train_losses)
        avg_dev = np.mean(dev_losses)
        history['train_loss'].append(avg_train)
        history['dev_loss'].append(avg_dev)
        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f} dev_loss={avg_dev:.4f}")
        if avg_dev < best_dev:
            best_dev = avg_dev
            best_epoch = epoch
            os.makedirs('outputs/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'outputs/checkpoints/fashion_best.pt')
            history['best_epoch'] = best_epoch
            print(f"New best epoch: {best_epoch+1}")
    # Save loss curves
    os.makedirs('outputs/plots', exist_ok=True)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['dev_loss'], label='dev')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('outputs/plots/fashion_loss_curves.png')
    plt.close()
    # Save history
    os.makedirs('outputs/tables', exist_ok=True)
    pd.DataFrame(history).to_csv('outputs/tables/fashion_history.csv', index=False)
    print(f"Best epoch: {best_epoch+1}")

if __name__ == "__main__":
    main()
