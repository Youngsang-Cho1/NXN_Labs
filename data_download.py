from datasets import load_dataset
from PIL import Image

dataset = load_dataset("owj0421/polyvore", split="data")

print('\n[Data Structure]')
print(dataset)

print(f'\n[Column Names]: {dataset.column_names}')

first_item = dataset[0]
print("\n[First Data Example]")
print(first_item)

print(f'\n[Image type]: {type(first_item["image"])}')
print(f'[Image size]: {first_item["image"].size}')
