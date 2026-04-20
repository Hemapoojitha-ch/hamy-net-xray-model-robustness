import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121Binary(pretrained=True).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_auroc, val_acc = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val AUROC: {val_auroc:.4f}, Val Acc: {val_acc:.4f}")
