import os
import random
import re
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        """
        A simple tokenizer that splits text into words.
        """
        text = text.lower()
        # Find all word characters
        return re.findall(r'\b\w+\b', text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_dict, img_keys, vocab, transform=None):
        self.root_dir = root_dir
        self.captions_dict = captions_dict
        self.img_keys = img_keys  # This is the list of image keys for the split
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, index):
        img_id = self.img_keys[index]
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        captions = self.captions_dict[img_id]
        # Use a random caption for each image during training
        caption = random.choice(captions)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption.extend(self.vocab.numericalize(caption))
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

def load_captions_and_split(captions_file, test_split=0.2, val_split=0.1, random_seed=42):
    print("Loading and splitting data using basic Python...")
    captions_dict = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            img_id, caption = parts[0], ','.join(parts[1:])
            if img_id not in captions_dict:
                captions_dict[img_id] = []
            captions_dict[img_id].append(caption)

    # Remove header if it was read as a key
    if 'image' in captions_dict:
        del captions_dict['image']

    image_names = list(captions_dict.keys())
    random.seed(random_seed)
    random.shuffle(image_names)
    
    test_size = int(len(image_names) * test_split)
    val_size = int(len(image_names) * val_split)
    
    test_keys = image_names[:test_size]
    val_keys = image_names[test_size : test_size + val_size]
    train_keys = image_names[test_size + val_size :]

    print(f"Total images: {len(image_names)}")
    print(f"Training samples: {len(train_keys)}")
    print(f"Validation samples: {len(val_keys)}")
    print(f"Test samples: {len(test_keys)}")
    
    return captions_dict, train_keys, val_keys, test_keys

def get_loader(root_folder, captions_dict, img_keys, vocab, transform, batch_size=32, shuffle=True, pin_memory=True):
    dataset = Flickr8kDataset(root_folder, captions_dict, img_keys, vocab, transform=transform)
    pad_idx = vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )
    return loader

def collate_fn(batch, pad_idx):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = pad_sequence(caps, batch_first=True, padding_value=pad_idx)
    return imgs, targets

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    captions_file = 'data/captions.txt'
    captions_dict, train_keys, _, _ = load_captions_and_split(captions_file)

    train_captions = [caption for key in train_keys for caption in captions_dict[key]]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_captions)
    print(f"Vocabulary Size: {len(vocab)}")
    
    train_loader = get_loader(
        root_folder='data/images',
        captions_dict=captions_dict,
        img_keys=train_keys,
        vocab=vocab,
        transform=transform
    )

    for imgs, captions in train_loader:
        print("Image shape:", imgs.shape)
        print("Captions shape:", captions.shape)
        break 