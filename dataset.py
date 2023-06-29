import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, seq_len=128, max_idx=2048, min_len=4):
        self.min_len = min_len
        self.sequence_length = seq_len
        self.max_idx = max_idx
        self.sequence = self.load_words(filename)
        self.start_indexes = self.get_indexes()

    def load_words(self, filename):
        with open(filename, 'r') as f:
            text = f.readlines()

        text = filter(lambda s: len(s) >= self.min_len, text)
        text = ('\n'.join(text) + '\n').replace('\n\n', '\n')
        text = text.encode('cp1251')

        return np.array([char for char in text])

    def get_indexes(self):
        indexes = [0]
        new_line_char_idx = '\n'.encode('cp1251')[0]

        for i in range(1, len(self.sequence) - self.sequence_length):
            if self.sequence[i - 1] == new_line_char_idx:
                indexes.append(i)

        advanced_indexes = []
        for index in indexes:
            while index + self.sequence_length < len(self.sequence):
                advanced_indexes.append(index)
                if self.sequence[index + self.sequence_length] == new_line_char_idx:
                    break

                index = index + self.min_len

        return list(sorted(set(advanced_indexes)))

    def __len__(self):
        return len(self.start_indexes)

    def __getitem__(self, index):
        start = self.start_indexes[index]
        chunk = self.sequence[start:start + self.sequence_length + 1]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)

        return train, target
