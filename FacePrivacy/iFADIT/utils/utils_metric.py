import numpy as np
class MyMetric:
    def __init__(self,metric_name):
        self.list = []
        self.metric_method = metric_name


    def _method(self, x1, x2):
        target = self.metric_method(x1, x2).cpu().detach().numpy()
        self.list.append(target)

    def _target_out(self):
        mean =  np.mean(self.list)
        return mean
    


if __name__ =="__main__":
    import torch
    from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio,MeanAbsoluteError
    device = "cuda:0"
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    metric_ssim = MyMetric(ssim)
    x1 = torch.rand((1,3,128,128), device=device)
    x2 = torch.rand((1,3,128,128), device=device)

    metric_ssim._method(x1, x2)
    print(f'metric_ssim list = {metric_ssim.list}')
    metric_ssim._method(x1, x2)
    print(f'metric_ssim list = {metric_ssim.list}')
    mean = metric_ssim._target_out()
    print(f'metric_ssim._target_out = {mean}')
