import os
import random
import numpy as np
# arr = [1,2,3,4,5]
# appe1 = []
# appe2 = []
# appe1.append(arr[:-1])
# appe2.append(arr[:1])

# print(appe1,appe2)
# print(arr)
# str = "Vu Tuan	Cong"; 
# str = str.split('\t')
# print (str)
# print (str.split(' ', 1 ))
# import os

# f = open('e15_cong','r')
# lines = f.readlines()

# for line in lines:
# 	print(line)
# __global_map = map #keep reference to the original map
# lmap = lambda func, *iterable: list(__global_map(func, *iterable)) # using "map" here will cause infinite recursion
# map = lmap
# x = [1, 2, 3]
# list = map(str, x) #test
# print(list)
# map = __global_map #restore the original map and don't do that again
# map(str, x)

# list = [20,16,10,5]
# random.shuffle(list)
# print("list sau khi bi xao tron la: ",list)
# random.shuffle(list)
# print("list sau khi bi xao tron la: ",list)

data = [[2,3],[3,4],[5,6],[1,3]]
shuffle_range = range(len(data))
print(shuffle_range)
shuffle_data = np.array(data)[shuffle_range]
print(shuffle_data)
