import torch
from sklearn.metrics import confusion_matrix as skl_cm
from sklearn.metrics import precision_recall_fscore_support as skl_score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def confusion_matrix(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        result = skl_cm(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return result

# def scores(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         result = skl_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
#     return result


