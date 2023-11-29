import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = tokenizer.vocab_size

        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.read()

        self.tokens = tokenizer.encode(self.data)

    def __len__(self):
        return len(self.tokens) - self.max_length

    def __getitem__(self, idx):
        # Input sequence is the context
        input_sequence = self.tokens[idx: idx + self.max_length]
        input_sequence = torch.tensor([input_sequence])

        # Target sequence is the next token
        target_sequence = self.tokens[idx + self.max_length]
        target_sequence = torch.tensor([target_sequence])

        return input_sequence, target_sequence
