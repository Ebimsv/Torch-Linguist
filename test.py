from torchtext.datasets import WikiText2

data_path = "path/to/save/dataset"

# Download the Wikitext-2 dataset
train_dataset, valid_dataset, test_dataset = WikiText2(root=data_path)