from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.Descriptors import FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3 \
    , ExactMolWt, HeavyAtomMolWt, MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge \
    , MinPartialCharge, NumRadicalElectrons, NumValenceElectrons
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3, CalcKappa1, CalcKappa2, CalcKappa3 \
    , CalcLabuteASA, CalcNumAliphaticCarbocycles, CalcNumAliphaticHeterocycles \
    , CalcNumAliphaticRings, CalcNumAmideBonds, CalcNumAromaticCarbocycles \
    , CalcNumAromaticHeterocycles, CalcNumAromaticRings, CalcNumAtomStereoCenters \
    , CalcNumBridgeheadAtoms, CalcNumHBA, CalcNumHBD, CalcNumHeteroatoms, CalcNumHeterocycles \
    , CalcNumLipinskiHBA, CalcNumLipinskiHBD, CalcNumRings, CalcNumRotatableBonds \
    , CalcNumSaturatedCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedRings \
    , CalcNumSpiroAtoms, CalcNumUnspecifiedAtomStereoCenters, CalcTPSA
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


chemdata = []
chemanswer = []
#with open("/home/choi/docking/mlp_test/smiles-grid_score.txt", "r") as chem:
with open("/home/choi/choipc/project/ReLeaSE_edit/docking/ucsf/mlp_test/smiles-grid_score.txt", "r") as chem:
#with open("/home/choi/choipc/project/ReLeaSE_edit/docking/ucsf/smiles-part/smiles-part9735", "r") as chem:
    Clines = chem.readlines()
    for Cline in Clines:
        # print(Cline)
        Csl = Cline.split('\t')
        # print(Csl)
        chemdata.append(Csl[0])  # smiles
        chemanswer.append(float(Csl[1]))  # grid_score
    # print(chemdata)


def Rdkit(smiles, answer):  # smiles, grid_score
    ms = []  # rdkit class로 smiles 변환
    for smile in smiles:
        _m = Chem.MolFromSmiles(smile)
        if _m:
            ms.append(_m)
    # print(ms)

    # rd = [[NumRadicalElectrons(m) for m in ms], [CalcFractionCSP3(m) for m in ms]
    #     , [0.01 * CalcLabuteASA(m) for m in ms], [CalcNumAliphaticCarbocycles(m) for m in ms]
    #     , [CalcNumAliphaticHeterocycles(m) for m in ms]
    #     , [CalcNumAmideBonds(m) for m in ms], [CalcNumAromaticCarbocycles(m) for m in ms]
    #     , [CalcNumAromaticHeterocycles(m) for m in ms]
    #     , [CalcNumHBA(m) for m in ms], [CalcNumHBD(m) for m in ms], [CalcNumHeteroatoms(m) for m in ms]
    #     , [CalcNumHeterocycles(m) for m in ms], [CalcNumLipinskiHBA(m) for m in ms]
    #     , [CalcNumLipinskiHBD(m) for m in ms], [CalcNumRings(m) for m in ms]
    #     , [CalcNumRotatableBonds(m) for m in ms], [0.1 * CalcTPSA(m) for m in ms]]
    Topol = [Chem.RDKFingerprint(x) for x in ms]
    # MACC = [MACCSkeys.GenMACCSKeys(x) for x in ms]
    finger = []
    an = []
    for i in range(len(ms)):
        Top = ",".join(Topol[i].ToBitString())
        # MA = ",".join(MACC[i].ToBitString())
        # print(MA)
        # Top = list(Topol[i])
        # MA = list(MACC[i])


        all = "{}".format(Top).split(",")
        all = [float(x) for x in all]

        finger.append(all)
        an.append(float(answer[i]))

    min_an = min(an)
    max_an = max(an)

    new_an = []
    for ii in range(len(ms)):
        new_answer = ((an[ii] - min_an) / (max_an - min_an))
        new_an.append(float(new_answer))

    return torch.tensor(finger, device=device), torch.tensor(new_an, device=device)


import random


def department(finger, answer, cut=0.2):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(finger)):
        atmp = random.random()
        if atmp >= cut:
            train_x.append(finger[i])
            train_y.append(answer[i])
        if atmp < cut:
            test_x.append(finger[i])
            test_y.append(answer[i])
    return train_x, train_y, test_x, test_y


device = torch.device("cuda")
# print(torch.cuda.is_available(), device)
train_x, train_y, test_x, test_y = department(chemdata, chemanswer)
x_train, y_train = Rdkit(train_x, train_y)
x_test, y_test = Rdkit(test_x, test_y)
length = x_train.size(1)

train = TensorDataset(x_train, y_train)
loader = DataLoader(train, batch_size=1024, shuffle=True)
test = TensorDataset(x_test, y_test)
loader2 = DataLoader(test, batch_size=1024, shuffle=True)

# print(train_x)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 1900)
        self.bn2 = nn.BatchNorm1d(1900)
        self.fc2 = nn.Linear(1900, 700)
        self.bn3 = nn.BatchNorm1d(700)
        self.fc3 = nn.Linear(700, 100)
        self.bn4 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 1)
        self.dr1 = nn.Dropout(0.1)
    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.dr1(x)
        y = self.fc4(x)
        pred = y
        return pred


model = Net().cuda()
# print(model)
# 오차함수
criterion = nn.MSELoss()

# 최적화 담당
optimizer = optim.SGD(model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',factor = 0.99)

# 학습

for epoch in range(3000):
    train_loss = 0
    test_loss = 0
    for x_train, y_train in loader:
        x_train, y_train = Variable(x_train), Variable(y_train)

        # 경사 초기화
        optimizer.zero_grad()

        # 순전파
        output = model(x_train)

        # output = model(x_test)
        # pred_data = output.cpu().data.numpy()
        # pred_data = [item for sublist in pred_data for item in sublist]

        # 오차
        loss = criterion(output, y_train)

        # 역전파 계산
        loss.backward()

        # 가중치 업데이트
        # scheduler.step(loss)
        optimizer.step()

        # 총 오차 업데이트
        train_loss += loss.data.item()

    # if (epoch + 1) % 50 == 0:
    #     for param_group in optimizer.param_groups:
    #         lr = param_group['lr']
    #     print(epoch + 1, train_loss, lr)

    #eval
    for x_test, y_test in loader2:
        x_test, y_test = Variable(x_test), Variable(y_test)
        pred_data = model(x_test)
        losses = criterion(pred_data, y_test)
        test_loss += losses.data.item()

        #cuda_tensor -> numpy
        pred_data = pred_data.cpu().data.numpy()
        pred_data = [item for sublist in pred_data for item in sublist]
        y_test_data = y_test.cpu().data.numpy()


    if (epoch + 1) % 50 == 0:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print(epoch + 1, test_loss, lr)

with open('./sgd5_pred_test.txt', 'w') as pred_:
    for i in range(len(pred_data)):
        pred_data[i], y_test_data[i]
        pred_.write("{0}\t{1}\n".format(pred_data[i], y_test_data[i]))

model_path = ('./lr_SGD5_MLP_L4.h5')
torch.save(model, model_path)

