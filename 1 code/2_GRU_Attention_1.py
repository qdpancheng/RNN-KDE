"""
    使用 GRU 对光伏发电功率提前 5min 预测
    根据偏相关图使用过去 4 个时刻的光伏功率及天气记录值作为输入，预测未来 5min 的光伏功率值

    note: 2_GRU_Attention_1.py
        仿照LSTM-attention论文中的构造方法，构造基于带有注意力机制的GRU的预测方法
"""
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt

torch.manual_seed(1)

TORCH_GPU = 1           # 是否使用GPU进行运算标志
GPU_NUM = 0             # GPU号确定, 现服务器包含两块GPU, 对应的GPU号分别为0, 1

INPUT_SIZE = 6          # 输入维数
HIDDEN_SIZE = 30        # 隐藏层神经元个数
#TIME_STEP = 5           # 时间序列长度
NUM_LAYERS = 5          # 隐藏层个数
SEQ = 4                 # 通过光伏功率的偏相关图确定输入时间序列的长度为4

# 定义LSTM网络
class myGRU(nn.Module):
    def __init__(self, Hidden_Size, Num_Layers):
        super(myGRU, self).__init__()
        self.Hidden_Size = Hidden_Size
        self.Num_layer = Num_Layers
        self.gru = nn.GRU(input_size=INPUT_SIZE,
                          hidden_size=self.Hidden_Size,
                          num_layers=self.Num_layer,
                          batch_first=True)
        self.attn = nn.Linear(INPUT_SIZE * SEQ, INPUT_SIZE * SEQ)    # 采用全连接层自动计算 t时刻 的输入与 前几个时刻的输入的相似性
        self.attn1 = nn.Linear(INPUT_SIZE * SEQ, SEQ)
        #self.out1 = nn.Linear(in_features=self.Hidden_Size, out_features=self.Hidden_Size)
        self.out = nn.Linear(in_features=self.Hidden_Size, out_features=1)

    def forward(self, input):
        r_out, hn = self.gru(input)   # r_out of shape [batch, seq_len, hidden_size]
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        # 计算 时刻t的 [天气变量, 光伏功率] 与其他时刻的 [天气变量, 光伏功率] 的相关性
        a = torch.cat([input[:, 0, :], input[:, 1, :], input[:, 2, :], input[:, 3, :]], dim=1)
        a = sigmoid(self.attn(a))
        atten_weight = F.softmax((self.attn1(a)), dim=1)
        #atten_weight = F.softmax(self.attn(torch.cat([input[:, 0, :], input[:, 1, :], input[:, 2, :], input[:, 3, :]], dim=1)), dim=1)
        w1 = atten_weight[:, 0]
        w2 = atten_weight[:, 1]
        w3 = atten_weight[:, 2]
        w4 = atten_weight[:, 3]
        w = torch.cat((w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1), w4.unsqueeze(1)), 1)
        atten_applied = torch.bmm(w.unsqueeze(1), r_out)
        out = self.out(torch.relu(atten_applied.squeeze(1)))
        #out = self.out(sigmoid(out))
        return out


if __name__ == '__main__':
    filePath = os.path.dirname(os.getcwd()) + r'/2 data'
    # print(filePath)
    # print(os.listdir(filePath))
    data = pd.read_excel(filePath + r'/data_20140401_20160531.xlsx')
    # print(data.count())
    # print(data.iloc[:, 3:5].columns)
    # 数据预处理--进行数据归一化 Standardize features by removing the mean and scaling to unit variance
    data_temp = np.zeros((222336, 5))  # 本次实验使用5个天气变量值，故data_temp的第二个维度设置为5
    data_temp = data.iloc[:, 3: -1].values  # 取出天气变量进行归一化处理 去掉风向，因离群点较多
    ss = StandardScaler()
    data_temp_std = ss.fit_transform(data_temp)
    data.iloc[:, 3: -1] = data_temp_std

    # 选择 5:00 -- 19:00 的光伏功率数据 (168)
    # 实际预测时间段 5:30 -- 19:00
    # 443 + 61 + 268 = 772天(去掉了数据缺失的天)  solar power + 5个天气变量 = 6  去掉风向特征（离群点太多）
    data_processed = np.zeros((772, 168, 6))
    for ii in range(772):
        temp = data.iloc[288 * ii + 60: 288 * ii + 228, 2: -1].values
        data_processed[ii, :, :] = temp

    print(data_processed.shape)
    train = data_processed[:443, :, :]
    val = data_processed[443: 504, :, :]
    test = data_processed[504: 772, :, :]

    # pytorch中设置LSTM中的 batch_first=True, input和output tensor --> (batch, seq, feature)
    # 对训练集而言，batch=357, seq : 将 05:30 -- 19:00 的数据构造sequence, 根据偏相关图得到seq=5, feature=5 (第0列为功率值，第1-5列为气象特征)
    # 对验证集而言，batch=30,
    # 对测试集而言，batch=31,
    # seq为4时，根据过去四个时间点的值预测未来的值
    # 重要！ 根据SEQ确定 jj + 2中的数字2
    train_processed = np.zeros((443 * 162, SEQ + 1, 6))  # 05:30 -- 19:00 共162个点(待预测点个数--一天产生的样本)
    for ii in range(443):
        for jj in range(162):
            train_processed[ii * 162 + jj, :, :] = train[ii, jj + 2: jj + 3 + SEQ,
                                                   :]  # 从05:00开始,取过去4个值进行预测,预测从05：30开始，故jj + 2

    val_processed = np.zeros((61 * 162, SEQ + 1, 6))
    for ii in range(61):
        for jj in range(162):
            val_processed[ii * 162 + jj, :, :] = val[ii, jj + 2: jj + 3 + SEQ, :]

    test_processed = np.zeros((268 * 162, SEQ + 1, 6))
    for ii in range(268):
        for jj in range(162):
            test_processed[ii * 162 + jj, :, :] = test[ii, jj + 2: jj + 3 + SEQ, :]

    print(train_processed.shape)
    print(val_processed.shape)
    print(test_processed.shape)
    # 对训练集、验证集、测试集进行数据处理，得到X_train, Y_train, X_val, Y_val, X_test, Y_test --X (batch, seq, feature)
    # 训练集
    X_train = np.zeros((train_processed.shape[0], SEQ, 6))  # feature = 6, 除了5个天气变量外，将光伏功率也考虑在内
    Y_train = np.zeros((train_processed.shape[0], 1))
    for ii in range(train_processed.shape[0]):
        X_train[ii, :, :] = train_processed[ii, :SEQ, :]
        Y_train[ii, :] = train_processed[ii, SEQ, 0]
    # 验证集
    X_val = np.zeros((val_processed.shape[0], SEQ, 6))
    Y_val = np.zeros((val_processed.shape[0], 1))
    for ii in range(val_processed.shape[0]):
        X_val[ii, :, :] = val_processed[ii, :SEQ, :]
        Y_val[ii, :] = val_processed[ii, SEQ, 0]
    # 测试集
    X_test = np.zeros((test_processed.shape[0], SEQ, 6))
    Y_test = np.zeros((test_processed.shape[0], 1))
    for ii in range(test_processed.shape[0]):
        X_test[ii, :, :] = test_processed[ii, :SEQ, :]
        Y_test[ii, :] = test_processed[ii, SEQ, 0]

    if (TORCH_GPU == 0):
        X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
        X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor)
        # Y_val = torch.from_numpy(Y_val).type(torch.FloatTensor)
        # Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor)
    else:
        X_train = torch.from_numpy(X_train).type(torch.FloatTensor).cuda(GPU_NUM)
        X_val = torch.from_numpy(X_val).type(torch.FloatTensor).cuda(GPU_NUM)
        X_test = torch.from_numpy(X_test).type(torch.FloatTensor).cuda(GPU_NUM)
        Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).cuda(GPU_NUM)
        # Y_val = torch.from_numpy(Y_val).type(torch.FloatTensor).cuda(GPU_NUM)
        # Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor)

    # ---------------------- 以下为超参数确定过程 ----------------------------------------------
    # 采用 Grid Search 的方法进行超参数的确定
    HIDDEN_SIZE = [30 + 10 * ii for ii in range(1)]
    NUM_LAYERS = [1 + ii for ii in range(1)]
    optimal_index = [0, 0, 0, 0]
    optimal_value = 1000
    for qqq, Hidden_Size in enumerate(HIDDEN_SIZE):
        for www, Num_layer in enumerate(NUM_LAYERS):
            print(Hidden_Size, Num_layer)
            if (TORCH_GPU == 0):
                gru = myGRU(Hidden_Size=Hidden_Size, Num_Layers=Num_layer)
            else:
                gru = myGRU(Hidden_Size=Hidden_Size, Num_Layers=Num_layer)
                gru = nn.DataParallel(gru.cuda())
            #print(gru)
            # 定义优化器
            optimizer = torch.optim.Adam(gru.parameters(), lr=0.002, weight_decay=1e-4)
            # 定义损失函数
            if (TORCH_GPU == 0):
                loss_function = nn.MSELoss()
            else:
                loss_function = nn.MSELoss().cuda()

            for epoch in range(1):
                # print('epoch is {}'.format(epoch))
                optimizer.zero_grad()
                prediction = gru(X_train)
                loss = loss_function(Y_train, prediction)
                loss.backward()
                optimizer.step()
                # if (TORCH_GPU == 0):
                #     #loss_save.append(loss.item())
                #     print(loss.item())  # 将tensor值变为python中的数值
                # else:
                #     #loss_save.append(loss.cpu().item())
                #     print(loss.cpu())
            pred_val = gru(X_val)
            if (TORCH_GPU == 0):
                pred_val = pred_val.detach().numpy()
            else:
                pred_val = pred_val.cpu().detach().numpy()
            Y_val = np.reshape(Y_val, (-1))
            pred_val = np.reshape(pred_val, (-1))
            error_RMSE = np.sqrt(np.mean(np.square(Y_val - pred_val)))
            if optimal_value > error_RMSE:
                optimal_value = error_RMSE
                optimal_index[0] = qqq
                optimal_index[1] = www
    print(optimal_index, HIDDEN_SIZE[optimal_index[0]], NUM_LAYERS[optimal_index[1]])
    print(optimal_value)
    # ---------------------- 以上为超参数确定过程 ----------------------------------------------

    # ==================== 以下为使用超参数确定模型并进行训练过程 ========================================
    if (TORCH_GPU == 0):
        gru = myGRU(Hidden_Size=80, Num_Layers=3)
    else:
        gru = myGRU(Hidden_Size=80, Num_Layers=3).cuda(GPU_NUM)
    print(gru)

    # 定义优化器
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.002, weight_decay=1e-4)
    # 定义损失函数
    if (TORCH_GPU == 0):
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.MSELoss().cuda(GPU_NUM)

    loss_save = []
    for epoch in range(1500):
        #print('epoch is {}'.format(epoch))
        optimizer.zero_grad()
        prediction = gru(X_train)
        loss = loss_function(Y_train, prediction)
        loss.backward()
        optimizer.step()
        if (TORCH_GPU == 0):
            loss_save.append(loss.item())
            #print(loss.item())   # 将tensor值变为python中的数值
        else:
            loss_save.append(loss.cpu().item())
            #print(loss.cpu())


    loss_csv = pd.DataFrame(loss_save)
    loss_csv.to_csv('aaloss_save_2_GRU_Attention_final.csv')

    # 保存完整模型
    torch.save(gru, 'aa2_GRU_Attention_final.pkl')

    pred_test = gru(X_test)
    if (TORCH_GPU == 0):
        pred_test = pred_test.detach().numpy()
    else:
        pred_test = pred_test.cpu().detach().numpy()
    # 将真实值与预测值进行保存
    pred = pd.DataFrame()
    Y_test = np.reshape(Y_test, (-1))
    pred['true'] = pd.Series(Y_test)
    pred_test = np.reshape(pred_test, (-1))
    pred['test'] = pd.Series(pred_test)
    pred.to_csv('aapred_value_2_GRU_Attention_final.csv')

    print(pred_test.shape)
    pred = np.reshape(pred_test, (-1))
    Y_test = np.reshape(Y_test, (-1))
    error_RMSE = np.sqrt(np.mean(np.square(Y_test[:9882] - pred[:9882])))
    print(error_RMSE)
    # plt.plot(pred, 'r')
    # plt.plot(Y_test)
    # plt.title('LSTM forecast')
    # plt.show()


