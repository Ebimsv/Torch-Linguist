import torchtext

# Set the path where you want to save the dataset
data_path = "./dataset"

# Download the Wikitext-2 dataset
train_dataset, valid_dataset, test_dataset = torchtext.datasets.WikiText2(root=data_path)

# Print the number of examples in each split
print(f"Number of training examples: {len(train_dataset.examples)}")
print(f"Number of validation examples: {len(valid_dataset.examples)}")
print(f"Number of test examples: {len(test_dataset.examples)}")