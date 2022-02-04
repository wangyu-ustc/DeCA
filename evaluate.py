import time
import multiprocessing
from multiprocessing import Process, Manager
import functools
import pickle
import numpy as np
import torch
import math
import os

def test_all_users(model, batch_size, item_num, test_data_pos, user_pos, top_k, device='cuda'):
    predictedIndices = []
    GroundTruth = []
    model.to(device)
    for u in test_data_pos:
        batch_num = item_num // batch_size
        batch_user = torch.Tensor([u] * batch_size).long().to(device)
        st, ed = 0, batch_size
        for i in range(batch_num):
            batch_item = torch.Tensor([i for i in range(st, ed)]).long().to(device)
            pred = model(batch_user, batch_item).detach().cpu()
            if i == 0:
                predictions = pred
            else:
                predictions = torch.cat([predictions, pred], 0)
            st, ed = st + batch_size, ed + batch_size
        ed = ed - batch_size
        batch_item = torch.Tensor([i for i in range(ed, item_num)]).long().to(device)
        batch_user = torch.Tensor([u] * (item_num - ed)).long().to(device)
        pred = model(batch_user, batch_item).detach().cpu()
        if batch_num > 0:
            predictions = torch.cat([predictions, pred], 0)
        else:
            predictions = pred
        test_data_mask = [0] * item_num
        if u in user_pos:
            for i in user_pos[u]:
                test_data_mask[i] = -9999
        predictions = predictions + torch.Tensor(test_data_mask).float()
        _, indices = torch.topk(predictions, top_k[-1])
        indices = indices.cpu().numpy().tolist()
        predictedIndices.append(indices)
        GroundTruth.append(test_data_pos[u])
    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR


def test_CDAE(model, test_batch_size, user_pos, top_k, test_data_pos, user_num, item_num, train_matrix, device='cuda'):
    model.to(device)
    predictedIndices = []
    GroundTruth = []
    # input_matrix = torch.zeros(user_num, item_num)
    eval_users = torch.tensor(list(test_data_pos.keys()))
    test_matrix = torch.zeros(len(eval_users), item_num)
    for u_idx, u in enumerate(test_data_pos):
        for i in test_data_pos[u]:
            test_matrix[u_idx, i] = 1

    num_data = len(eval_users)
    num_batches = int(np.ceil(num_data / test_batch_size))
    perm = torch.arange(num_data)

    for b in range(num_batches):
        batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
        user_idx = eval_users[batch_idx].to(device)
        if isinstance(train_matrix, torch.Tensor):
            input_matrix = train_matrix[user_idx].to(device)
        else:
            input_matrix = _convert_sp_mat_to_sp_tensor(train_matrix[batch_idx]).to_dense().to(device)
        batch_pred_matrix = model(user_idx, input_matrix)
        batch_pred_matrix.masked_fill(input_matrix.to(device).bool(), float('-inf'))
        for idx, (u_idx, u) in enumerate(zip(batch_idx, eval_users[batch_idx])):
            predictions = batch_pred_matrix[idx].detach().cpu()
            _, indices = torch.topk(predictions, top_k[-1])
            indices = indices.cpu().numpy().tolist()
            predictedIndices.append(indices)
            GroundTruth.append(test_data_pos[u.item()])

    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def compute_acc(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))

    return precision, recall, NDCG, MRR
