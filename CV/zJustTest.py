import os

import torch
import pandas as pd

device = 'cpu'

pred = torch.tensor([[1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.]]).to(device)

label = torch.tensor([[1., 1.],
                       [2., 1.],
                       [3., 1.],
                       [4., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.],
                       [1., 1.]]).to(device)

def tracking_sample(pred, label, sample_path):
    dummy = torch.tensor([[999], [999], [999], [999], [999], [999], [999], [999], [999]])
    result_tensor = torch.cat((pred.cpu(), dummy.cpu()), dim=1)
    result_df = pd.DataFrame(result_tensor.cpu().numpy())
    file_exists = os.path.isfile(sample_path)
    if file_exists:
        result_df = result_df.replace(999, '->')
        saved_df = pd.read_csv(sample_path, header=None)
        updated_df = pd.concat([saved_df, result_df], axis=1)
        updated_df.to_csv(sample_path, header=False, index=False)
    else:
        dummy = torch.tensor([[888], [888], [888], [888], [888], [888], [888], [888], [888]])
        result_tensor = torch.cat((label.cpu(), dummy.cpu(), result_tensor.cpu()), dim=1)
        result_df = pd.DataFrame(result_tensor.cpu().numpy())
        result_df = result_df.replace(999, '->')
        result_df = result_df.replace(888, 'vs')
        result_df.to_csv(sample_path, header=False, index=False)


tracking_sample(pred, label, './save/test.csv')
