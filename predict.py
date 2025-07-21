import torch
import torchvision.transforms as transforms
from PIL import Image
from models.model import CNNtoRNN
from utils.data_loader import Vocabulary, load_captions_and_split
import argparse

def predict(image_path, model_path, vocab):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Model parameters (should match training)
    embed_size = 512
    hidden_size = 512
    num_layers = 2 # Match the training script
    vocab_size = len(vocab)
    
    # Load the trained model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # Generate caption using greedy search
    print("Generating caption...")
    caption_tokens = model.caption_image(image_tensor, vocab)
    
    # Clean and join the tokens
    if caption_tokens[0] == '<SOS>':
        caption_tokens = caption_tokens[1:]
    if caption_tokens[-1] == '<EOS>':
        caption_tokens = caption_tokens[:-1]

    result = ' '.join(caption_tokens)
    print("Generated Caption:", result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a caption for an image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights.')
    
    args = parser.parse_args()

    # We need to build the same vocabulary as used in training
    print("Building vocabulary...")
    captions_file = 'data/captions.txt'
    captions_dict, train_keys, _, _ = load_captions_and_split(captions_file)
    train_captions = [caption for key in train_keys for caption in captions_dict[key]]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_captions)
    print("Vocabulary built.")
    
    predict(args.image, args.model_path, vocab) 