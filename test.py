import argparse
import os.path
import torch

from model import Model
from tools import evaluate

device = torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model.pth')
args = parser.parse_args()

input_size = 2 ** 8
model = Model(input_size=input_size, embedding_size=input_size*2, device=device).to(device)

text = ''
while text != 'exit':
    text = input()

    if os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model))

    with torch.no_grad():
        print(evaluate(model, start_text=text))
