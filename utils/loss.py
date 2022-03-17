import torch


class ComputeLoss(object):
    # Compute losses
    def __init__(self, model):
        super(ComputeLoss, self).__init__()
        self.BCE_cls = torch.nn.CrossEntropyLoss()  # torch.nn.BCEWithLogitsLoss()

    def __call__(self, predictions, targets):
        # predictions = torch.max(predictions, dim=1).indices.float()
        loss = self.BCE_cls(predictions, targets)
        return loss
