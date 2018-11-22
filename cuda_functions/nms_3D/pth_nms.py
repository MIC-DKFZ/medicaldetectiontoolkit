import torch
from ._ext import nms


def nms_gpu(dets, thresh):
  """
  dets has to be a tensor
  """

  scores = dets[:, -1]
  order = scores.sort(0, descending=True)[1]
  dets = dets[order].contiguous()

  keep = torch.LongTensor(dets.size(0))
  num_out = torch.LongTensor(1)
  nms.gpu_nms(keep, num_out, dets, thresh)
  return order[keep[:num_out[0]].cuda()].contiguous()


def nms_cpu(dets, thresh):

  dets = dets.cpu()
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  z1 = dets[:, 4]
  z2 = dets[:, 5]
  scores = dets[:, 6]
  areas = (x2 - x1 +1) * (y2 - y1 +1) * (z2 - z1 +1)
  order = scores.sort(0, descending=True)[1]

  keep = torch.LongTensor(dets.size(0))
  num_out = torch.LongTensor(1)
  nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

  return keep[:num_out[0]]

