import numpy as np
import sys
import os
import torch
import pandas as pd
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 40)
        self.layer3 = nn.Linear(40, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x))  # To check with the loss function
        return x

# funkcja usuwająca wiersze zawierające platformę "Stadia"
def delete_stadia(games):
    index_list = []
    for i in range(0, len(games["platform"])):
        try:
            if games["platform"][i] == " Stadia":
                index_list.append(i)
        except:
            continue
    games.drop(index_list, inplace=True)
    return games.reset_index()

# funkcja usuwająca wiersze zawierające "tbd" w kolumnie "user_review"
def delete_tbd(games):
    index_list = []
    for i in range(0, len(games["platform"])):
        try:
            if games["user_review"][i] == 'tbd':
                index_list.append(i)
        except:
            continue
    games.drop(index_list, inplace=True)
    return games.reset_index()

def delete_PC(games):
    index_list = []
    for i in range(0, len(games["platform"])):
        try:
            if games["platform"][i] == " PC":
                index_list.append(i)
        except:
            continue
    games.drop(index_list, inplace=True)
    return games.reset_index()

# funkcja zmieniająca kolumnę "user_review" ze stringa na numeric
def user_review_to_numeric(games):
    games["user_review"] = pd.to_numeric(games["user_review"])
    return games

# funkcja normalizująca wartości w kolumnie "meta_score" i "user_review"
def normalization(games):
    games['meta_score'] = games['meta_score'] / 100.0
    games['user_review'] = games['user_review'] / 10.0
    return games

# old - 0
# mid - 1
# new - 2

def platform_to_number(games):
    for i in range(0, len(games["platform"])):

        if games["platform"][i] == " PlayStation":
            games["platform"][i] = 0
        elif games["platform"][i] == " PlayStation 2":
            games["platform"][i] = 0
        elif games["platform"][i] == " PlayStation 3":
            games["platform"][i] = 1
        elif games["platform"][i] == " PlayStation 4":
            games["platform"][i] = 2
        elif games["platform"][i] == " PlayStation 5":
            games["platform"][i] = 2
        elif games["platform"][i] == " PlayStation Vita":
            games["platform"][i] = 1
        elif games["platform"][i] == " Xbox":
            games["platform"][i] = 0
        elif games["platform"][i] == " Xbox 360":
            games["platform"][i] = 1
        elif games["platform"][i] == " Xbox Series X":
            games["platform"][i] = 2
        elif games["platform"][i] == " Nintendo 64":
            games["platform"][i] = 0
        elif games["platform"][i] == " GameCube":
            games["platform"][i] = 0
        elif games["platform"][i] == " DS":
            games["platform"][i] = 0
        elif games["platform"][i] == " 3DS":
            games["platform"][i] = 1
        elif games["platform"][i] == " Wii":
            games["platform"][i] = 0
        elif games["platform"][i] == " Wii U":
            games["platform"][i] = 1
        elif games["platform"][i] == " Switch":
            games["platform"][i] = 2
        elif games["platform"][i] == " PC":
            dt = datetime.strptime(games["release_date"][i], '%B %d, %Y')
            if (dt.year == 1995 or dt.year == 1996 or dt.year == 1997 or dt.year == 1998
                or dt.year == 1999 or dt.year == 2000 or dt.year == 2001 or dt.year == 2002
                    or dt.year == 2003 or dt.year == 2004 or dt.year == 2005):
                games["platform"][i] = 0
            if (dt.year == 2006 or dt.year == 2007 or dt.year == 2008 or dt.year == 2009
                or dt.year == 2010 or dt.year == 2011 or dt.year == 2012 or dt.year == 2013
                    or dt.year == 2014 or dt.year == 2015 or dt.year == 2016):
                games["platform"][i] = 1
            if (dt.year == 2017 or dt.year == 2018 or dt.year == 2019
                or dt.year == 2020 or dt.year == 2021):
                games["platform"][i] = 2

            # games["platform"][i] = 0
        elif games["platform"][i] == " Dreamcast":
            games["platform"][i] = 0
        elif games["platform"][i] == " Game Boy Advance":
            games["platform"][i] = 0
        elif games["platform"][i] == " PSP":
            games["platform"][i] = 1
        elif games["platform"][i] == " Xbox One":
            games["platform"][i] = 2

    return games

def remove_list(games):
    for i in range(0, len(games)):
        games['platform'][i] = games['platform'][i][0]
        games['release_date'][i] = games['release_date'][i][0]
        games['meta_score'][i] = games['meta_score'][i][0]
        games['user_review'][i] = games['user_review'][i][0]
    return games

platform = pd.read_csv('all_games.train.csv', sep=',', usecols=[1], header=None).values.tolist()
release_date = pd.read_csv('all_games.train.csv', sep=',', usecols=[2], header=None).values.tolist()
meta_score = pd.read_csv('all_games.train.csv', sep=',', usecols=[4], header=None).values.tolist()
user_review = pd.read_csv('all_games.train.csv', sep=',', usecols=[5], header=None).values.tolist()

games_test = {'platform': platform,
     'release_date': release_date,
     'meta_score': meta_score,
     'user_review': user_review}
games_test = pd.DataFrame(games_test)

games_test = remove_list(games_test)
games_test = platform_to_number(games_test)
games_test = delete_stadia(games_test)
games_test = delete_tbd(games_test)
games_test = user_review_to_numeric(games_test)
games_test = normalization(games_test)

labels_test_g = pd.DataFrame(games_test["platform"], dtype=np.int64)
labels_test_g = labels_test_g.to_numpy()
features_test_g = {'meta_score': games_test['meta_score'],
              'user_review': games_test['user_review']}
features_test_g = pd.DataFrame(features_test_g, dtype=np.float64)
features_test_g = features_test_g.to_numpy()

# Training
model = torch.load("games_model.pkl")

# Prediction
x_test = Variable(torch.from_numpy(features_test_g)).float()
pred = model(x_test)

pred = pred.detach().numpy()

accuracy = accuracy_score(labels_test_g, np.argmax(pred, axis=1))

pred = pd.DataFrame(pred)

predicted = []
expected = []

for i in range(0, len(x_test)):
    predicted.append(np.argmax(model(x_test[i]).detach().numpy(), axis=0))
    expected.append(labels_test_g[i])

for i in range(0, len(expected)):
    expected[i] = expected[i][0]

precision, recall, fscore, support = precision_recall_fscore_support(expected, predicted, average="micro")

res = f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F-score: {fscore}"

with open('metrics.txt', 'a+') as f:
    f.write(str(accuracy) + '\n')

with open('metrics.txt') as f:
    accuracy_val = [float(line) for line in f if line]
    builds = list(range(1, len(accuracy_val) + 1))

plt.xlabel('Build')
plt.ylabel('Accuracy')
plt.plot(builds, accuracy_val, label='Accuracy')
plt.legend()
plt.show()
plt.savefig('metrics.png')