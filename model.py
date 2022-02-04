import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                    model, dropout=0.0, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
                user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
                item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                    a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                            self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                            self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                            self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                            self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                        self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = torch.sigmoid(self.predict_layer(concat))
        return prediction.view(-1)

    def user_embedding(self, user):
        if self.model == 'GMF':
            return self.embed_user_GMF(user)
        elif self.model == 'NeuMF-end':
            return torch.cat([self.embed_user_GMF(user), self.embed_user_MLP(user)], dim=1)
    def item_embedding(self, item):
        if self.model == 'GMF':
            return self.embed_item_GMF(item)
        elif self.model == 'NeuMF-end':
            return torch.cat([self.embed_item_GMF(item), self.embed_item_MLP(item)], dim=1)

class MF(nn.Module):
    def __init__(self, user_num, item_num, K0):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(user_num, K0)
        self.item_embedding = nn.Embedding(item_num, K0)
        self._init_weight_()
    def forward(self, user, item):
        # user: batch_size
        # item: batch_size
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        # return torch.sigmoid(torch.diag(torch.mm(user_embedding, item_embedding.transpose(0,1))))
        p = torch.sigmoid(torch.bmm(user_embedding.unsqueeze(1), item_embedding.unsqueeze(1).transpose(1,2))).reshape(-1)
        return p

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
