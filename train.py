from GPT2Trainer.dataset import CustomDataset
from GPT2Model.model import GPT2
from GPT2Trainer.trainer import Trainer, TrainerConfig
from transformers import GPT2Tokenizer

if __name__ == "__main__":

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load the dataset
    dataset = CustomDataset(file_path="./data/lotr.txt", tokenizer=tokenizer)

    # Initialize the model
    model = GPT2()

    # Initialize the trainer
    config = TrainerConfig()
    trainer = Trainer(model=model, dataset=dataset, config=config)

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()