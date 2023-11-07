# Created by Baole Fang at 10/31/23
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import pickle
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler


def load(path: str, idx: int = 0, subset: int = 1000):
    sampler = RandomUnderSampler()
    data = np.load(path)
    X = []
    Y = []
    for x, y in zip(data['X'], data['Y']):
        ys = y.split(';')
        if idx < len(ys):
            X.append(x)
            Y.append(ys[idx])
    X = X[:subset]
    Y = Y[:subset]
    pX = []
    pY = []
    for x1, y1 in zip(X, Y):
        for x2, y2 in zip(X, Y):
            pX.append(np.concatenate([x1, x2]))
            pY.append(1 if y1 == y2 else 0)
    return sampler.fit_resample(np.array(pX), np.array(pY))


def main(args):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    X, y = load(args.input, args.column, args.number)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=args.seed)
    print(f'X_train: {X_train.shape} y_train: {y_train.shape}')
    print(f'X_test: {X_test.shape} y_test: {y_test.shape}')
    model = xgboost.XGBClassifier(n_jobs=os.cpu_count())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).astype(int)
    report = classification_report(y_test, y_pred)
    path = os.path.join(args.output, os.path.basename(args.input).split('.')[0], args.model)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(path, 'metrics.txt'), 'w') as f:
        f.write(report)
    df = pd.DataFrame(cm)
    df.to_csv(os.path.join(path, 'confusion.csv'), sep=' ', index=False, header=False)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(os.path.join(path, 'confusion.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the classification problem')
    parser.add_argument('-i', '--input', help='path of the data', type=str, required=True)
    parser.add_argument('-c', '--column', help='column/index of label', type=int, default=0)
    parser.add_argument('-n', '--number', help='number of samples', type=int, default=1000)
    parser.add_argument('-t', '--test', help='test set size', type=float, default=0.2)
    parser.add_argument('-o', '--output', help='path of output model', type=str, default='outputs')
    parser.add_argument('-m', '--model', help='model name', type=str, default='xgboost')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
