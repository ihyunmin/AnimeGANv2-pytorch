import torch
import torch.nn as nn
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable

# random = torch.randint(1,10,(2,1,4,4))

# print(random.shape)
# assert random.shape == (2,1,4,4), 'hey'
# print(random)
# mean_random = torch.mean(random.float(),(2,3))
# print(mean_random)
# print(mean_random.shape)

# loss = torch.nn.MSELoss()

# Tensor = torch.FloatTensor
# valid = Variable(Tensor(random.shape).fill_(1.0), requires_grad=False)
# print(valid)

# test_loss = loss(random, valid)
# print(test_loss)

# bceloss = torch.nn.BCELoss()
# print(torch.sigmoid(random/10))
# print(random/10)
# print(bceloss(torch.sigmoid(random/10),valid))


# class test():
#     def __init__(self):
#         super(test,self).__init__()
    
#     @staticmethod
#     def add(a, b):
#         return a+b

# a = 3
# b = 5

# k = test()
# print(k.add(a,b))
# a = [1,2,3,1,7]
# b = [2,3,4,5,6]
# at = torch.Tensor(a)
# bt = torch.Tensor(b)

# loss = nn.L1Loss()
# loss2 = loss(at,bt)
# print(loss2)


# tf.enable_eager_execution()

# m = nn.ReflectionPad2d(2)
# input = torch.arange(3, dtype=torch.float).reshape(1,1,1,3)
# print(input)
# # target = m(input)

# paddings = (0,0,3,2,1,2)
# target = torch.nn.functional.pad(input, paddings)
# print(target.shape)
# print(target)


# t = tf.random.uniform((1,1,1,3),minval=0,maxval=10, dtype=tf.int64)
# # print(t.numpy())

# x = tf.pad(t, [[0,0],[1,2],[3,2],[0,0]])
# print(x.shape)

# kernel = 3
# channels = 64
# x = torch.normal(mean=0, std=0.02, size=(12,258,258, 3))
# w = torch.normal(mean=0, std=0.02, size=(kernel, kernel, x.shape[-1], channels))

# print(x.shape)

# print(x.shape)
# sn = torch.nn.utils.spectral_norm(nn.Conv2d(3,32,3,stride=1))
# x = sn(x)
# print(x.shape)

# print(type(x))
# print(x.shape)