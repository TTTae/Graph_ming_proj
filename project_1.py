import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoders import Encoder
from Aggregators import Aggregator

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse
import os
import time


parser = argparse.ArgumentParser(description='GCN Recommendation')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs')
args = parser.parse_args()


class GCNRec(nn.Module):
    def __init__(self, enc_u, enc_v):
        super(GCNRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v = enc_v
        self.emb_dim = enc_u.embed_dim
        self.linear_layer = nn.Linear(embed_dim*2, 1)

    def forward(self, nodes_u, nodes_v):
        output_u = self.enc_u_history.forward(nodes_u)
        output_v = self.enc_v_history.forward(nodes_v)
        output_uv = torch.cat((output_u, output_v), 1)
        output = self.linear_layer(output_uv)
        return output


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

embed_dim = args.embed_dim
dir_data = './data/my_dataset'
path_data = dir_data + ".pickle"
data_file = open(path_data, 'rb')
u_item_lists, u_item_r_lists, v_lists, vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
    data_file)
# u_item_lists: items each user purchase
# u_item_r_lists: corresponding rates for u_item_lists
# v_list: users who purchase item i
# vr_lists: corresponding rates for v_list
# we do not use social list and rating list here


print("Loading data")
trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                         torch.FloatTensor(test_r))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

num_users = u_item_lists.__len__()
num_items = v_lists.__len__()
num_ratings = ratings_list.__len__()

u2e = nn.Embedding(num_users, embed_dim).to(device)
v2e = nn.Embedding(num_items, embed_dim).to(device)
r2e = nn.Embedding(num_ratings, embed_dim).to(device)

agg_u = Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
enc_u = Encoder(u2e, embed_dim, u_item_lists, v_lists, agg_u, cuda=device, uv=True)
enc_u = enc_u.to(device)

agg_v = Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
enc_v = Encoder(v2e, embed_dim, u_item_lists, v_lists, agg_v, cuda=device, uv=False)
enc_v = enc_v.to(device)


### try

#
# loss_func = nn.MSELoss()
# idx, data = next(enumerate(train_loader, 0))
# batch_nodes_u, batch_nodes_v, labels_list = data
# gcn_rec = MyModel(enc_u_history, enc_v_history).to(device)
# res = gcn_rec(batch_nodes_u.to(device), batch_nodes_v.to(device))
# res = res.view(-1)
# loss = loss_func(res, labels_list.to(device))
# print(loss.item())


gcn_rec = GCNRec(enc_u, enc_v).to(device)
optimizer = torch.optim.RMSprop(gcn_rec.parameters(), lr=args.lr, alpha=0.9)
loss_func = nn.MSELoss()

best_rmse = 100.0
best_mae = 100.0
endure_count = 0

### training
start_time = time.time()
for epoch in range(1, args.epochs + 1):
    gcn_rec.train()
    cum_loss = 0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        res = gcn_rec(batch_nodes_u.to(device), batch_nodes_v.to(device))
        res = res.view(-1)
        loss = loss_func(res, labels_list.to(device))
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, cum_loss / 100, best_rmse, best_mae))
            cum_loss = 0.0

    ### test
    expected_rmse, mae = test(gcn_rec, device, test_loader)

    ### early stop
    if best_rmse > expected_rmse:
        best_rmse = expected_rmse
        best_mae = mae
        endure_count = 0
    else:
        endure_count += 1
    print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

    if endure_count > 5:
        break

end_time = time.time()
print("Time consuming: ", end_time-start_time)


