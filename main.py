import torch
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

#군집 개수
K_DIV = 4
#최대 반복
Iteration = 100
#사진 경로
PATH_DATA = 'Car_sales.csv'
#GPU
DEVICE = 'cuda'

def initial_centroids(input):
    '''현재 y스펙트럼이 4 이상 안올라감'''
    #first - 무작위 선택
    first = input[0, :]

    #second - 1번 점에서 가장 먼 점
    second = torch.square(input - first).cuda(device=DEVICE)
    second = torch.sum(abs(second), dim=1).cuda(device=DEVICE)
    max = second.max(dim=0)
    second = input[int(max[1])]

    #third - 1, 2번 점에서 가장 먼 점
    third = torch.square((input - first) + (input - second)).cuda(device=DEVICE)
    third = torch.sum(abs(third), dim=1).cuda(device=DEVICE)
    max = third.max(dim=0)
    third = input[int(max[1])]

    #fourth - 선행 점들과 가장 먼 점
    fourth = torch.square((input - first) + (input - second) + (input - third)).cuda(device=DEVICE)
    fourth = torch.sum(abs(fourth), dim=1).cuda(device=DEVICE)
    max = fourth.max(dim=0)
    fourth = input[int(max[1])]

    result = torch.stack([first, second, third, fourth]).cuda(device=DEVICE)
    return result

def assign_cluster(input, center):
    expended_vector = torch.unsqueeze(input, 0).cuda(device=DEVICE)
    expended_center = torch.unsqueeze(center, 1).cuda(device=DEVICE)
    distances = torch.sum(torch.square(torch.sub(expended_vector, expended_center)), 2).cuda(device=DEVICE)
    min = torch.argmin(distances, 0).cuda(device=DEVICE)
    return min.cuda(device=DEVICE)

def recompute_centroids(X, Y, data_amount, K):
    # torch.zeros(1,1).scatter_add(dim, idx, src)
    X = X.reshape([data_amount, 2])

    #X, Y좌표 각각의 중심점을 위한 배열 분리
    XX, XY = [], []
    for i in X:
        XX.append(i[0])
        XY.append(i[1])

    XX, XY = torch.tensor(XX).reshape([data_amount, 1]).float().cuda(device=DEVICE), torch.tensor(XY).reshape([data_amount, 1]).float().cuda(device=DEVICE)
    Y = Y.reshape([data_amount, 1]).long().cuda(device=DEVICE)

    #X, Y좌표 각각의 중심축 별 데이터간 거리 적산 거리
    dummy_X = torch.zeros(data_amount, K).float().cuda(device=DEVICE).scatter_add(1, Y, XX).cuda(device=DEVICE)
    dummy_Y = torch.zeros(data_amount, K).float().cuda(device=DEVICE).scatter_add(1, Y, XY).cuda(device=DEVICE)

    sumX = torch.sum(dummy_X, dim=0, keepdim=True).cuda(device=DEVICE)
    sumY = torch.sum(dummy_Y, dim=0, keepdim=True).cuda(device=DEVICE)

    count = torch.zeros(data_amount, K).float().cuda(device=DEVICE).scatter_add(1, Y, torch.ones_like(XX)).cuda(device=DEVICE)
    count = torch.sum(count, dim=0, keepdim=True).cuda(device=DEVICE)

    #ZeroDIV 방지
    for i in range(len(count[0])):
        if count[0][i] == 0:
            count[0][i] += 1

    resultX = (sumX // count).cuda(device=DEVICE)
    resultY = (sumY // count).cuda(device=DEVICE)
    tmp = torch.cat((resultX, resultY), dim=0).cuda(device=DEVICE)

    result = []
    for col in range(len(tmp[0])):
        result.append([tmp[0][col], tmp[1][col]])
    result = torch.tensor(result).reshape([4, 2])
    return result

if __name__ == '__main__':
    #데이터 셋 업로드 및 전처리
    sales = []
    engine = []
    data = []
    data_csv = pd.read_csv(PATH_DATA, header=None)

    data_amount = 0
    for row_idx, row in data_csv.iterrows():
        if row_idx != 0:
            data_amount += 1
            sales.append(float(row[2]))
            engine.append(float(row[6]))
            data.append([float(row[2]), float(row[6])])

    print('총 데이터 개수\t', data_amount)
    print('데이터 읽어오기 완료')

    '''초기 데이터 시각화'''
    '''plt.scatter(sales, engine)
    plt.xlabel('Sales_in_thousandEngine_size')
    plt.ylabel('Engine_size')
    plt.show()
    plt.pause(1)
    plt.close()
    print('데이터 시각화 완료')'''

    '''반복과 수렴 확인'''
    cycle, converged, centroids_history = 0, False, []

    '''초기 데이터셋 할당 및 중심잡기'''
    X = torch.tensor(data).cuda(device=DEVICE)
    centroids = initial_centroids(X)

    while not converged:
        '''비지도 학습'''
        Y = assign_cluster(X, centroids)
        centroids = recompute_centroids(X, Y, data_amount, K_DIV)
        centroids_history.append(centroids)
        if len(centroids_history) > 1 and torch.equal(centroids_history[-1], centroids_history[-2]):
            converged = True
        print('iteration', len(centroids_history), 'centroids\n', centroids)
    print('final centroids\n', centroids)
    print('분석 완료')

    '''결과 시각화'''
    for i in range(len(X)):
        plt.scatter(X[i][0].detach().cpu().clone().numpy(), X[i][1].detach().cpu().clone().numpy(), c='blue', label='Data')
    for i in range(len(centroids)):
        plt.scatter(centroids[i][0], centroids[i][1], s=300, c='purple', label='Centroids')
    plt.xlabel('Sales_in_thousandEngine_size')
    plt.ylabel('Engine_size')
    plt.title('K-means result')
    plt.legend(loc='upper right')
    plt.show()
    plt.pause(1)
    plt.close()
    print('분석결과 시각화 완료')

