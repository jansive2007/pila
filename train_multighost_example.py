from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from multi_ghost.dataset import build_dataloader


class SimpleImitationNet(nn.Module):
    def __init__(self, in_channels: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        stack_size=args.stack_size,
        shuffle=True,
        num_workers=0,
    )

    model = SimpleImitationNet(in_channels=3 * args.stack_size, n_actions=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for frames, actions, _meta in loader:
            frames = frames.to(device)
            actions = actions.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss={epoch_loss:.6f}")

    torch.save(model.state_dict(), "multighost_model.pt")
    print("Saved model to multighost_model.pt")


if __name__ == "__main__":
    main()
