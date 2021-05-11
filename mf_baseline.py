import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse
import os
import time


class MF_baseline(nn.Module):
    def __init__(self, embed_dim, num_users, num_items, cuda="cpu"):
        super(MF_baseline, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.device = cuda

    def forward(self, nodes_u, nodes_v):
        e_u = self.user_emb.weight[nodes_u]
        e_i = self.item_emb.weight[nodes_v]
        num_train_nodes = e_u.shape[0]
        num_feat = e_u.shape[1]
        res = torch.bmm(e_u.view(num_train_nodes, 1, num_feat), e_i.view(num_train_nodes, num_feat, 1))
        return res


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model(test_u, test_v)
            val_output = val_output.view(-1)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae



parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

embed_dim = args.embed_dim
dir_data = './data/my_dataset'
path_data = dir_data + ".pickle"
data_file = open(path_data, 'rb')
history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
    data_file)

trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                         torch.FloatTensor(test_r))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

print("loading data")
num_users = history_u_lists.__len__()
num_items = history_v_lists.__len__()
num_ratings = ratings_list.__len__()


#
# loss_func = nn.MSELoss()
# idx, data = next(enumerate(train_loader, 0))
# batch_nodes_u, batch_nodes_v, labels_list = data
# mf_baseline = MF_baseline(embed_dim, num_users, num_items).to(device)
# res = mf_baseline(batch_nodes_u.to(device), batch_nodes_v.to(device))
#
# res = res.view(-1)
# print(res)

# loss = loss_func(res, labels_list.to(device))
# print(loss.item())


mf_baseline = MF_baseline(embed_dim, num_users, num_items).to(device)

optimizer = torch.optim.RMSprop(mf_baseline.parameters(), lr=args.lr, alpha=0.9)
loss_func = nn.MSELoss()

best_rmse = 100.0
best_mae = 100.0
endure_count = 0

### training
start_time = time.time()
for epoch in range(1, args.epochs + 1):
    mf_baseline.train()
    cum_loss = 0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        res = mf_baseline(batch_nodes_u.to(device), batch_nodes_v.to(device))
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
    expected_rmse, mae = test(mf_baseline, device, test_loader)

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

