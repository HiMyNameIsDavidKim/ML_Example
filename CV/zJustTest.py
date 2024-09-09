import os

import torch
import pandas as pd
from matplotlib import pyplot as plt


def tracking_sample(self, pred, label, sample_path):
    dummy_vs = torch.tensor([[888], [888], [888], [888], [888], [888], [888], [888], [888]])
    dummy_arrow = torch.tensor([[999], [999], [999], [999], [999], [999], [999], [999], [999]])
    result_tensor = torch.cat((label.cpu(), dummy_vs.cpu(), pred.cpu()), dim=1)
    file_exists = os.path.isfile(sample_path)
    if not file_exists:
        result_df = pd.DataFrame(result_tensor.cpu().numpy())
        result_df = result_df.replace(888, 'vs')
        result_df.to_csv(sample_path, header=False, index=False)
    else:
        result_tensor = torch.cat((dummy_arrow.cpu(), result_tensor.cpu()), dim=1)
        result_df = pd.DataFrame(result_tensor.cpu().numpy())
        result_df = result_df.replace(888, 'vs')
        result_df = result_df.replace(999, '->')
        saved_df = pd.read_csv(sample_path, header=None)
        updated_df = pd.concat([saved_df, result_df], axis=1)
        updated_df.to_csv(sample_path, header=False, index=False)


if __name__ == '__main__':
    file_name = f'./data/tracking_sample_{6}.csv'
    tracker(file_name)
