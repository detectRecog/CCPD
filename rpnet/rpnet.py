# Compared to fh0.py
# fh02.py remove the redundant ims in model input
from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from load_data import *
from roi_pooling import roi_pooling_ims
from torch.optim import lr_scheduler


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to the input file")
ap.add_argument("-n", "--epochs", default=10000,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5,
                help="batch size for train")
ap.add_argument("-se", "--start_epoch", required=True,
                help="start epoch for train")
ap.add_argument("-t", "--test", required=True,
                help="dirs for test")
ap.add_argument("-r", "--resume", default='111',
                help="file for re-train")
ap.add_argument("-f", "--folder", required=True,
                help="folder to store model")
ap.add_argument("-w", "--writeFile", default='fh02.out',
                help="file for output")
args = vars(ap.parse_args())

wR2Path = './wR2/wR2.pth2'
use_gpu = torch.cuda.is_available()
print (use_gpu)

numClasses = 7
numPoints = 4
classifyNum = 35
imgSize = (480, 480)
# lpSize = (128, 64)
provNum, alphaNum, adNum = 38, 25, 35
batchSize = int(args["batchsize"]) if use_gpu else 2
trainDirs = args["images"].split(',')
testDirs = args["test"].split(',')
modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'
storeName = modelFolder + 'fh02.pth'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)

epochs = int(args["epochs"])
#   initialize the output file
if not os.path.isfile(args['writeFile']):
    with open(args['writeFile'], 'wb') as outF:
        pass


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x


class fh02(nn.Module):
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path))
            # self.wR2 = self.wR2.cuda()
        # for param in self.wR2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).cuda(), requires_grad=False)
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).cuda(), requires_grad=False)
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).cuda(), requires_grad=False)

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).cuda(), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        # input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
        # rois = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]


epoch_start = int(args["start_epoch"])
resume_file = str(args["resume"])
if not resume_file == '111':
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print ("fail to load existed model! Existing ...")
        exit(0)
    print ("Load existed model! %s" % resume_file)
    model_conv = fh02(numPoints, numClasses)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = fh02(numPoints, numClasses, wR2Path)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.cuda()

print(model_conv)
print(get_n_params(model_conv))

criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

dst = labelFpsDataLoader(trainDirs, imgSize)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=8)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)


def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)


def eval(model, test_dirs):
    count, error, correct = 0, 0, 0
    dst = labelTestDataLoader(test_dirs, imgSize)
    testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=8)
    start = time()
    for i, (XI, labels, ims) in enumerate(testloader):
        count += 1
        YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        #   compare YI, outputY
        try:
            if isEqual(labelPred, YI[0]) == 7:
                correct += 1
            else:
                pass
        except:
            error += 1
    return count, correct, error, float(correct) / count, (time() - start) / count


def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, Y, labels, ims) in enumerate(trainloader):
            if not len(XI) == batchSize:
                continue

            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            Y = np.array([el.numpy() for el in Y]).T
            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(Y).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(Y), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            try:
                fps_pred, y_pred = model(x)
            except:
                continue

            # Compute and print loss
            loss = 0.0
            loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
            loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
            for j in range(7):
                l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                loss += criterion(y_pred[j], l)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            try:
                lossAver.append(loss.data[0])
            except:
                pass

            if i % 50 == 1:
                with open(args['writeFile'], 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                torch.save(model.state_dict(), storeName)
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        model.eval()
        count, correct, error, precision, avgTime = eval(model, testDirs)
        with open(args['writeFile'], 'a') as outF:
            outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        torch.save(model.state_dict(), storeName + str(epoch))
    return model


model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
