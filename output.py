import torch
from network import data_loader


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def classification(model):
    total = 0
    correct = 0
# drop_last switched to False to keep all samples
    _,test_loader = data_loader(False)
    with torch.no_grad():
      model.eval()
      for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        # forward pass
        test_spk, _ = model(data.view(data.size(0), -1))
        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
