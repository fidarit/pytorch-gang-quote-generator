import argparse
import os.path
import random
import signal
import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from dataset import Dataset
from tools import evaluate


def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            drop_last=True, shuffle=True, generator=torch.Generator(device=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    sheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    loss_avg = []
    for epoch in range(args.max_epochs):
        for train, target in tqdm(dataloader):
            optimizer.zero_grad()
            train = train.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)

            output, _ = model(train)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

            loss.backward()
            optimizer.step()
            loss_avg.append(loss.item())

        print({'epoch': epoch, 'loss_mean': np.mean(loss_avg), 'loss_median': np.median(loss_avg)})
        sheduler.step()
        with torch.no_grad():
            print(evaluate(model, start_text='Я '))
            print(' ')
            print(evaluate(model, start_text='Жи ши '))
            print(' ')
            print(evaluate(model, start_text='Лучше '))
            print(' ')
        torch.save(model.state_dict(), args.model)


if __name__ == "__main__":
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)

    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('CUDA using')

    print('Press Ctrl+C to exit')
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit())

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64*8)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--dataset', type=str, default='data/dataset_main.txt')
    parser.add_argument('--cpu', type=bool, default=False)
    args = parser.parse_args()

    device = torch.device('cuda') if not args.cpu and torch.cuda.is_available() else torch.device('cpu')

    input_size = 2**8
    dataset = Dataset(args.dataset, max_idx=input_size, seq_len=args.sequence_length)
    model = Model(input_size=input_size, embedding_size=input_size*2).to(device)

    if os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model))

    train(dataset, model, args)
