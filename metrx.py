# The modlue is used to calculate performance metrics on a confusion 
# matrix passed in as a torch.FloatTensor
# Developed with Python 3.7.3

import torch

class Error(Exception):
   """Base class for other exceptions"""
   pass
class NotTensor(Error):
   """Raised when input is not a torch.tensor"""
   pass


def accuracy(matrix):
  if matrix.type() == 'torch.FloatTensor':
    acc = torch.sum(matrix.diag()) / torch.sum(matrix)
    return acc
  else:
    raise NotTensor

def per_class_accuracy(matrix):

  if matrix.type() == 'torch.FloatTensor':
    acc = matrix.diag() / matrix.sum(1)
    return acc
    
  else:
    raise NotTensor

def sensitivity(matrix, class_index):
  if matrix.type() == 'torch.FloatTensor':
    NUM = matrix[class_index, class_index]
    DEN = torch.sum(matrix[:,class_index])
    return NUM / DEN
  else:
    raise NotTensor

def precision(matrix, class_index):
  if matrix.type() == 'torch.FloatTensor':
    NUM = matrix[class_index, class_index]
    DEN = torch.sum(matrix[class_index,:])
    return NUM / DEN
  else:
    raise NotTensor

def number_false_negative(matrix, class_index):
  if matrix.type() == 'torch.FloatTensor':
    return torch.sum(matrix[:,class_index]) - matrix[class_index, class_index]
  else:
    raise NotTensor

def number_false_positive(matrix, class_index):
  if matrix.type() == 'torch.FloatTensor':
    return torch.sum(matrix[class_index,:]) - matrix[class_index, class_index]
  else:
    raise NotTensor


