import os
import sys
import copy
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from LightGCN import LightGCN
import scipy.sparse as sp
from CDAE import CDAE
import numpy as np
import evaluate
from model import MF, NCF

def log(x):
    return torch.log(x + 1e-5)

def drop_rate_schedule(iteration):

    drop_rate = np.linspace(0, 0.2**1, 30000)
    if iteration < 30000:
        return drop_rate[iteration]
    else:
        return 0.2

class TrainModel():
    def __init__(self, param, model, user_num, item_num):
        self.param = param
        self.model = model
        self.user_num = user_num
        self.item_num = item_num
        self.seed = param['seed']
        self.top_k = param['top_k']
        self.method = param['method']
        self.C_1 = param.get('C_1', 1000)
        self.C_2 = param.get('C_2', 10)
        self.NSR = self.param.get("NSR", 1)
        self.alpha = self.param.get("alpha", 0.5)
        self.eval_freq = param.get("eval_freq", 500)
        self.denoise_type = param.get('denoise_type', 'both')
        self.early_stop = param.get("early_stop", False)
        self.pretrain_early_stop = param.get("pretrain_ealry_stop", True)
        self.early_stop_rounds = param.get("early_stop_rounds", 2)
        self.emb_dim = param.get("emb_dim", 32)
        self.h_model = param.get("h_model", 'MF')
        self.lr = param.get('lr', 0.001)

        if self.h_model == 'GMF' or self.h_model == 'NeuMF-end':
            self.h_model_1 = NCF(user_num, item_num, self.emb_dim, 3, model=self.h_model)
            self.h_model_2 = NCF(user_num, item_num, self.emb_dim, 3, model=self.h_model)
        else:
            self.h_model_1 = MF(user_num=user_num, item_num=item_num, K0=self.emb_dim)
            self.h_model_2 = MF(user_num=user_num, item_num=item_num, K0=self.emb_dim)

        if self.param['method'] == 'DeCAp':
            self.aux_model = copy.deepcopy(model)
        else:
            self.aux_model = MF(user_num=user_num, item_num=item_num, K0=self.emb_dim)

        self.h_model_1_optim = optim.Adam(self.h_model_1.parameters(), lr=self.lr)
        self.h_model_2_optim = optim.Adam(self.h_model_2.parameters(), lr=self.lr)
        self.aux_model_optim = optim.Adam(self.aux_model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.print_config()

    def print_config(self):
        print(f"seed: {self.seed}")
        print(f"method: {self.method}")
        print(f"model: {self.param['model']}, dataset: {self.param['dataset']}")
        if self.method == 'DeCA' or self.method == 'DeCAp':
            print("h_model:", self.h_model)
            print("alpha:", self.alpha)
            print(f"C_1 used in DP: {self.C_1}, C_2 used in DN: {self.C_2}")
        print(f"epochs: {self.param['epochs']}, early_stop: {self.early_stop}, early_stop_rounds: {self.early_stop_rounds}")
        print(f"regularization lambda: {self.param['lambda0']}")
        print(f"negative sampling rate: {self.NSR}")

    def forward(self, model, user, pos_item, NSR=1):
        """
        :param user: user
        :param pos_item: positive items tensor: (bsz, )
        :return: positive scores and negative scores
        """
        neg_item = self.sample_neg_items(user)
        # if not isinstance(model, LightGCN):
        pos_prediction = model(user, pos_item)
        neg_prediction = model(user, neg_item)
        if NSR > 1:
            for i in range(NSR - 1):
                new_neg_item = self.sample_neg_items(user)
                neg_prediction = torch.cat([neg_prediction,
                            model(user, new_neg_item)], dim=0)
                neg_item = torch.cat([neg_item, new_neg_item], dim=0)

        return pos_prediction, neg_prediction, neg_item

    def sample_neg_items(self, user):
        neg_item = []
        for single_user in user:
            j = self.random_choice()
            while (single_user, j) in self.train_mat:
                j = self.random_choice()
            neg_item.append(j)
        neg_item = torch.tensor(neg_item).long().to(self.device)
        return neg_item

    def random_choice(self):
        return np.random.randint(self.item_num)


    def train(self, train_loader,
            valid_loader,
            test_data_pos,
            valid_data_pos,
            test_data_noisy,
            train_mat,
            user_pos,
    ):
        DP_NSR = self.param.get('DP_NSR', self.NSR)
        DN_NSR = self.param.get('DN_NSR', self.NSR)
        batch_size = self.param.get('batch_size', 2048)
        epochs = self.param['epochs']

        self.train_mat = train_mat
        self.valid_data_pos = valid_data_pos
        self.device=self.param.get("device", 'cuda')
        if self.method == 'DeCA' or self.method == 'DeCAp':
            self.h_model_1.to(self.device)
            self.aux_model.to(self.device)
            self.h_model_2.to(self.device)
        self.model.to(self.device)

        def _on_iteration_start():
            self.model.zero_grad()
            self.h_model_1_optim.zero_grad()
            self.aux_model_optim.zero_grad()
            self.h_model_2_optim.zero_grad()
            self.NSR = DP_NSR if self.denoise_type == 'DP' else DN_NSR

        def _on_iteration_end(count):
            if self.denoise_type != 'DN':
                self.h_model_1_optim.step()
            if self.denoise_type != 'DP':
                self.h_model_2_optim.step()
            if self.method == 'DeCA':
                self.aux_model_optim.step()
            self.optimizer.step()
            if self.param.get("iterative", True):
                self.denoise_type = 'DN' if self.denoise_type == 'DP' else 'DP'

        def RS_train(model, variational, optimizer, train_early_stop):
            count, last_loss = 0, 1e9
            ES_count = 0

            for epoch in range(self.param['epochs']):
                model.train()
                for user, item, label, noisy_or_not in train_loader:
                    _on_iteration_start()
                    user = user.to(self.device)
                    item = item.to(self.device)
                    label = label.float().to(self.device)
                    pos_prediction, neg_prediction, neg_item = self.forward(model, user, item, self.NSR)
                    if variational:
                        pos_loss = torch.mean(self.KL_loss(user, item, pos_prediction) \
                               - self.positive_loss_function(user, item, pos_prediction, count))

                        neg_loss = torch.mean(self.KL_loss(user.repeat(self.NSR), neg_item, neg_prediction) \
                                - self.negative_loss_function(user.repeat(self.NSR), neg_item, neg_prediction, count))

                        loss = pos_loss + neg_loss * self.NSR

                        reg_loss = 0
                        for param in model.parameters():
                            reg_loss += param.norm(2).pow(2)
                        reg_loss = 1 / 2 * reg_loss / float(self.user_num)

                        loss += reg_loss * self.param['lambda0']
                        loss.backward()
                        _on_iteration_end(count)
                    else:
                        loss = F.binary_cross_entropy(pos_prediction, label) \
                               + F.binary_cross_entropy(neg_prediction, torch.zeros_like(label).repeat(self.NSR))
                        reg_loss = 0
                        for param in model.parameters():
                            reg_loss += param.norm(2).pow(2)
                        reg_loss = 1/2 * reg_loss / float(self.user_num)
                        loss += reg_loss * self.param['lambda0']
                        loss.backward()
                        optimizer.step()

                    if count % 200 == 0 and count != 0:
                        print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))

                    if count % self.eval_freq == 0 and count != 0:
                        epoch_loss = self.eval(model, valid_loader, count)
                        print("Eval: epoch: {}, iter: {}, Eval loss:{}".format(epoch, count, epoch_loss))
                        if epoch_loss > last_loss:
                            ES_count += 1
                            if train_early_stop and ES_count >= self.early_stop_rounds:
                                return
                        last_loss = epoch_loss
                        model.train()
                    count += 1

        def CDAE_train(train_matrix, valid_data_pos, model, variational, optimizer, train_early_stop, debug=None):
            best_loss, count = 1e9, 0
            ES_count = 0
            valid_users = torch.tensor(list(valid_data_pos.keys()))
            valid_matrix = torch.zeros(len(valid_users), self.item_num)
            for u_idx, u in enumerate(valid_data_pos):
                for i in valid_data_pos[u]:
                    valid_matrix[u_idx, i] = 1

            num_training = train_matrix.shape[0]
            for epoch in range(epochs):
                num_batches = int(np.ceil(num_training / batch_size))
                perm = np.random.permutation(num_training)
                for b in range(num_batches):
                    optimizer.zero_grad()

                    if (b + 1) * batch_size >= num_training:
                        batch_idx = perm[b * batch_size:]
                    else:
                        batch_idx = perm[b * batch_size: (b + 1) * batch_size]

                    if isinstance(train_matrix, torch.Tensor):
                        batch_matrix = train_matrix[batch_idx]
                    else:
                        batch_matrix = self._convert_sp_mat_to_sp_tensor(train_matrix[batch_idx]).to_dense()

                    non_zero_user = torch.where(torch.sum(batch_matrix, dim=1) > 0)

                    batch_matrix = batch_matrix[non_zero_user].to(self.device)
                    batch_idx = batch_idx[non_zero_user]

                    batch_idx = torch.LongTensor(batch_idx).to(self.device)
                    pred_matrix = model(batch_idx, batch_matrix)

                    if not variational:
                        # cross_entropy
                        batch_user_id, item = torch.where(batch_matrix == 1)
                        batch_loss = F.binary_cross_entropy(pred_matrix[(batch_user_id, item)],
                                                    batch_matrix[(batch_user_id, item)], reduction='sum')
                        pos_number = len(batch_user_id)

                        batch_user_id, neg_item = torch.where(batch_matrix == 0)
                        neg_id = torch.tensor(random.sample(range(len(batch_user_id)), pos_number * self.NSR))
                        batch_user_id = batch_user_id[neg_id]
                        neg_item = neg_item[neg_id]

                        batch_loss += F.binary_cross_entropy(pred_matrix[(batch_user_id, neg_item)],
                                             batch_matrix[(batch_user_id, neg_item)], reduction='sum')

                        for param in model.parameters():
                            batch_loss += torch.norm(param) * self.param['lambda0']

                        batch_loss.backward()
                        optimizer.step()
                    else:
                        _on_iteration_start()
                        pos_gamma, neg_gamma = None, None
                        if self.method == 'DeCAp':
                            gamma_matrix = self.aux_model(batch_idx, batch_matrix).detach()

                        # find positive samples
                        batch_user_id, item = torch.where(batch_matrix == 1)
                        user = batch_idx[batch_user_id]
                        pos_prediction = pred_matrix[(batch_user_id, item)]
                        if self.method == 'DeCAp':
                            pos_gamma = gamma_matrix[(batch_user_id, item)]
                        loss = torch.mean(self.KL_loss(user, item, pos_prediction, pos_gamma) \
                           - self.positive_loss_function(user, item, pos_prediction, count))

                        pos_number = len(user)
                        # find negative samples
                        neg_batch_user_id, neg_item = torch.where(batch_matrix==0)

                        # neg_id = torch.randperm(len(batch_user_id))[: pos_number * self.NSR]
                        neg_id = torch.tensor(random.sample(range(len(neg_batch_user_id)), pos_number * self.NSR))
                        neg_batch_user_id = neg_batch_user_id[neg_id]
                        neg_item = neg_item[neg_id]

                        neg_user = batch_idx[neg_batch_user_id]
                        neg_prediction = pred_matrix[(neg_batch_user_id, neg_item)]

                        if self.method == 'DeCAp':
                            neg_gamma = gamma_matrix[(neg_batch_user_id, neg_item)]

                        loss += torch.mean(self.KL_loss(neg_user, neg_item, neg_prediction, neg_gamma) \
                               - self.negative_loss_function(neg_user, neg_item, neg_prediction, count))

                        for param in model.parameters():
                            loss += self.param['lambda0'] * torch.norm(param)

                        loss.backward()
                        _on_iteration_end(count)
                        batch_loss = loss.item()

                    if count % 50 == 0:
                        print('epoch [%d]: (%3d / %3d) loss = %.4f' % (epoch, b, num_batches, batch_loss))

                    if count % self.eval_freq == 0:
                        # using noisy data to evaluate our model
                        loss = self.eval_CDAE(model, valid_users, count, valid_matrix)
                        print("################### EVAL ######################")
                        print("epoch: {}, iter: {}, Eval loss:{}".format(epoch, count, loss))
                        if loss > best_loss:
                            ES_count += 1
                            if train_early_stop and ES_count >= self.early_stop_rounds:
                                return
                        best_loss = loss
                    count += 1

        if isinstance(self.model, CDAE):
            train_matrix = torch.zeros([self.user_num, self.item_num])
            # train_matrix = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
            for (u, i) in self.train_mat.keys():
                train_matrix[u, i] = 1.0

            if self.method == 'DeCAp':
                try:
                    self.aux_model = torch.load(os.path.join(self.param['folder'], f"{self.param['model']}_{self.param['dataset']}.ckpt")).to(self.device)
                    print("load pretrained model successfully")
                except:
                    print("pretrain model...")
                    CDAE_train(train_matrix, valid_data_pos, model=self.aux_model, variational=False, optimizer=self.aux_model_optim, train_early_stop=self.pretrain_early_stop)
                    torch.save(self.aux_model, os.path.join(self.param['folder'], f"{self.param['model']}_{self.param['dataset']}.ckpt"))
                    print("pretrain model done")
            CDAE_train(train_matrix, valid_data_pos, model=self.model, variational=(self.method in ['DeCA', 'DeCAp']), optimizer=self.optimizer, train_early_stop=self.early_stop)

        else:
            if self.method == 'DeCAp':
                # RS_pretrain()
                try:
                    self.aux_model = torch.load(os.path.join(self.param['folder'], f"{self.param['model']}_{self.param['dataset']}.ckpt")).to(self.device)
                    print("load pretrained model successfully")
                except:
                    print("pretrained model not found, train from scratch")
                    print("pretrain_model...")
                    if (self.param['model'] == 'GMF' or self.param['model'] == 'LightGCN') and self.param['dataset'] == 'electronics':
                        RS_train(self.aux_model, variational=0, optimizer=self.aux_model_optim,
                                 train_early_stop=False)
                    else:
                        RS_train(self.aux_model, variational=0, optimizer=self.aux_model_optim,
                                 train_early_stop=self.pretrain_early_stop)
                    torch.save(self.aux_model, os.path.join(self.param['folder'], f"{self.param['model']}_{self.param['dataset']}.ckpt"))
                    print("pretrain model done")
                # self.test(self.aux_model, test_data_pos, user_pos, top_k)
            RS_train(self.model, variational=(self.method in ['DeCA', 'DeCAp']), optimizer=self.optimizer, train_early_stop=self.early_stop)

        print("############################## Training End. ##############################")

        if self.param.get("save_model", False):
            if self.method is None:
                mode = 'BS'
            else:
                mode = self.method

            torch.save(self.model, os.path.join(self.param['folder'], f"{self.param['model']}_{self.param['dataset']}_{mode}_{self.seed}.ckpt"))

        if isinstance(self.model, CDAE):
            clean_precision, clean_recall, clean_NDCG, clean_MRR \
                = evaluate.test_CDAE(self.model, 2048, user_pos, self.top_k, test_data_pos, self.user_num, self.item_num, train_matrix, device='cuda')
            # noisy_precision, noisy_recall, noisy_NDCG, noisy_MRR \
            #     = evaluate.test_CDAE(self.model, 2048, user_pos, top_k, test_data_noisy, self.user_num, self.item_num, train_matrix)
        else:
            clean_precision, clean_recall, clean_NDCG, clean_MRR \
                = evaluate.test_all_users(self.model, 4096, self.item_num, test_data_pos, user_pos, self.top_k, device='cuda')
            # noisy_precision, noisy_recall, noisy_NDCG, noisy_MRR \
            #     = evaluate.test_all_users(self.model, 4096, self.item_num, test_data_noisy, user_pos, top_k, device='cuda')
        print("################### CLEAN TEST ######################")
        print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_recall[0], clean_recall[1],clean_recall[2],clean_recall[3]))
        print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_NDCG[0], clean_NDCG[1],clean_NDCG[2],clean_NDCG[3]))
        # print("################### NOISY TEST ######################")
        # print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(noisy_recall[0], noisy_recall[1], noisy_recall[2], noisy_recall[3]))
        # print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(noisy_NDCG[0], noisy_NDCG[1], noisy_NDCG[2], noisy_NDCG[3]))

        return clean_precision, clean_recall, clean_NDCG, clean_MRR


    def positive_loss_function(self, user, item, prediction, count):
        if self.denoise_type == 'DP':
            return log(self.h_model_1(user, item)) * (1 - prediction)
        elif self.denoise_type == 'DN':
            return log(self.h_model_2(user, item)) * prediction - self.C_2 * (1 - prediction)
        else:
            return log(1 - self.h_model_1(user, item)) * (1 - prediction) + \
                   log(1 - self.h_model_2(user, item)) * prediction

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def negative_loss_function(self, user, item, prediction, count):
        if self.denoise_type == 'DP':
            return log(1 - self.h_model_1(user, item)) * (1 - prediction) - prediction * self.C_1
        elif self.denoise_type == 'DN':
            return log(1 - self.h_model_2(user, item)) * prediction
        else:
            return log(1 - self.h_model_1(user, item)) * (1 - prediction) + \
                   log(1 - self.h_model_2(user, item)) * prediction

    def KL_loss(self, user, item, prediction, gamma=None):
        if self.method=='DeCAp' and gamma is None:
            p = self.aux_model(user, item).detach()
        elif gamma is None:
            p = self.aux_model(user, item)
        else:
            p = gamma
        loss = self.KL(p, prediction) * self.alpha + self.KL(prediction, p) * (1-self.alpha)
        return loss

    def eval(self, model, valid_loader, count):
        model.eval()
        epoch_loss = 0
        valid_loader.dataset.ng_sample()  # negative sampling
        for user, item, label, noisy_or_not in valid_loader:
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.float().to(self.device)

            prediction = model(user, item)
            loss = F.binary_cross_entropy(prediction, label)
            epoch_loss += loss.detach()
        return epoch_loss

    def eval_CDAE(self, model, valid_users, count, valid_matrix, eval_batch_size=2048):
        num_data = valid_matrix.shape[0]
        num_batches = int(np.ceil(num_data / eval_batch_size))

        loss = 0
        for b in range(num_batches):
            input_matrix = valid_matrix[b * eval_batch_size: (b + 1) * eval_batch_size].to(self.device)
            input_users = valid_users[b * eval_batch_size: (b + 1) * eval_batch_size].to(self.device)
            pred_matrix = model(input_users, input_matrix)
            loss += F.binary_cross_entropy(pred_matrix, input_matrix)
        return loss


    def KL(self, p1, p2):
        return p1 * log(p1) - p1 * log(p2) + \
               (1 - p1) * log(1 - p1) - (1 - p1) * log(1 - p2)

