import torch
from torchmetrics import Metric


class RMSE(Metric):
    full_state_update = False
    higher_is_better = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):

        diff = pred - target
        squared_error = diff ** 2

        self.sum_squared_error += squared_error.sum()

        self.total_elements += pred.numel()

    def compute(self):
    
        mse = self.sum_squared_error / self.total_elements
        return torch.sqrt(mse)
