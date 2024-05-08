import numpy as np

from ptflops import get_model_complexity_info

def count_ModelSize_byptflops(model, inputSize):
  # SUMMARY = summary(model, inputSize)

  macs, params = get_model_complexity_info(model, inputSize, as_strings=False,
                                           print_per_layer_stat=True, verbose=False)

  # print(f"TorchInfo summary : \n  {SUMMARY}")
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  return macs, params

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

