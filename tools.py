import numpy as np
import torch.nn.functional as F
import torch


def evaluate(model, prev_text='', start_text=' ', max_len=512, temp=0.3):
    model.eval()
    device = model.device
    idx_input = [char for char in (prev_text + start_text).encode('cp1251')]
    inp = torch.IntTensor([idx_input[0]]).view(-1, 1, 1)
    predicted_text = start_text
    hidden = None

    for idx in idx_input:
        _, hidden = model(inp.to(device), hidden)
        inp = torch.IntTensor([idx]).view(-1, 1, 1)

    for i in range(max_len):
        output, hidden = model(inp.to(device), hidden)
        p_next = F.softmax(output.view(-1) / temp, dim=-1).cpu().numpy()
        top_index = np.random.choice(model.input_size, p=p_next)
        inp = torch.IntTensor([top_index]).view(-1, 1, 1)
        predicted_char = bytes([top_index])
        predicted_char = predicted_char.decode('cp1251')

        if predicted_char == '\n':
            break

        predicted_text += predicted_char

    model.train()
    return predicted_text


def replace_all(text, old, new):
    while old in text:
        text = text.replace(old, new)

    return text
