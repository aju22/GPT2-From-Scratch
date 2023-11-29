from dataclasses import dataclass
import torch
import torch.nn as nn
from GPT2Model.model import GPT2
from GPT2Trainer.dataset import CustomDataset


@dataclass
class TrainerConfig:
    """
    Configuration class for the model trainer.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training.
        num_epochs (int): The number of training epochs.
        use_ddp (bool): Whether to use Distributed Data Parallel (DDP) for training.
        use_fsdp (bool): Whether to use Fully Sharded Data Parallelism (FSDP) for training.
        use_gpu (bool): Whether to use GPU for training.
    """

    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 1
    use_ddp: bool = False
    use_fsdp: bool = False
    use_gpu: bool = True


class Trainer:
    def __init__(self, model: GPT2, dataset: CustomDataset, config: TrainerConfig):

        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cpu')

        if self.config.use_gpu:
            assert torch.cuda.is_available(), "No GPU Found"
            self.device = torch.device('cuda')

        # Data loader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Wrap the model with FSDP if configured
        if self.config.use_fsdp:
            self.model = torch.distributed.fsdp.FullyShardedDataParallel(self.model)

        # Distributed setup if using DDP
        if self.config.use_ddp:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.model = nn.parallel.DistributedDataParallel(self.model)

    def train(self):
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                input_sequence, target_sequence = batch

                # Move data to device
                input_sequence, target_sequence = input_sequence.to(self.device), target_sequence.to(self.device)

                # Forward pass
                output = self.model(input_sequence)
                output_logits = output[:, -1, :]

                # Compute loss
                loss = self.criterion(output_logits, target_sequence)

                self.optimizer.zero_grad()
                loss.backward()

                # Update parameters using the optimizer
                with torch.no_grad():
                    self.optimizer.step()

                # Print statistics
                print(f'Epoch: {epoch + 1}/{self.config.num_epochs}, Loss: {loss.item()}')
