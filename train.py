import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # Import tqdm
from utils.data_loader import get_loader, load_captions_and_split, Vocabulary
from models.model import CNNtoRNN
import os

def train():
    # Hyperparameters
    embed_size = 512
    hidden_size = 512
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 80
    
    # Setup for Tensorboard
    writer = SummaryWriter("runs/flickr")

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory
    if not os.path.exists('models/weights'):
        os.makedirs('models/weights')

    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load Data
    captions_file = 'data/captions.txt'
    captions_dict, train_keys, val_keys, _ = load_captions_and_split(captions_file)

    # Initialize vocabulary and build it from the training captions
    train_captions = [caption for key in train_keys for caption in captions_dict[key]]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_captions)
    vocab_size = len(vocab)

    # Get data loaders
    train_loader = get_loader(
        root_folder='data/images',
        captions_dict=captions_dict,
        img_keys=train_keys,
        vocab=vocab,
        transform=transform,
        batch_size=64,
    )
    
    # Initialize model, loss, optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for imgs, captions in progress_bar:
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward prop
            outputs = model(imgs, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            # Backward prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar with the latest loss
            progress_bar.set_postfix(loss=loss.item())
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Average epoch loss", avg_epoch_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_epoch_loss:.4f}.")


    # Save the final model
    torch.save(model.state_dict(), f"models/weights/image_captioning_final.pth")
    writer.close()
    print("\nTraining complete. Final model saved as image_captioning_final.pth")

if __name__ == "__main__":
    train() 