import torch
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu

from models.model import CNNtoRNN
from utils.data_loader import Vocabulary, load_captions_and_split

def evaluate(model_path):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # --- Load Data and Vocabulary ---
    print("Loading data and vocabulary...")
    captions_file = 'data/captions.txt'
    captions_dict, _, _, test_keys = load_captions_and_split(captions_file)
    
    # We need to build the same vocabulary as used in training
    _, train_keys, _, _ = load_captions_and_split(captions_file)
    train_captions_list = [caption for key in train_keys for caption in captions_dict[key]]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_captions_list)
    print("Vocabulary built.")

    # --- Load Model ---
    # Model parameters (should match training)
    embed_size = 256
    hidden_size = 256
    num_layers = 2 # Match the training script
    vocab_size = len(vocab)
    
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    # --- Generate Captions for Test Set ---
    references = []
    hypotheses = []
    
    print("Generating captions for test set...")
    for key in tqdm(test_keys, desc="Evaluating"):
        # Get reference captions
        ref_captions = captions_dict[key]
        # Tokenize references
        tokenized_refs = [Vocabulary.tokenizer(c) for c in ref_captions]
        references.append(tokenized_refs)
        
        # Generate hypothesis caption
        img_path = f"data/images/{key}"
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        generated_caption_tokens = model.caption_image(image, vocab)
        
        # Remove <SOS> and <EOS> tokens
        if generated_caption_tokens[0] == '<SOS>':
            generated_caption_tokens = generated_caption_tokens[1:]
        if generated_caption_tokens[-1] == '<EOS>':
            generated_caption_tokens = generated_caption_tokens[:-1]
            
        hypotheses.append(generated_caption_tokens)

    # --- Calculate BLEU Scores ---
    print("\n--- Calculating BLEU Scores ---")
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("\n--- Overall Performance on Test Set ---")
    print(f"Corpus BLEU-1: {bleu1*100:.2f}")
    print(f"Corpus BLEU-2: {bleu2*100:.2f}")
    print(f"Corpus BLEU-3: {bleu3*100:.2f}")
    print(f"Corpus BLEU-4: {bleu4*100:.2f}")
    print("---------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the Image Captioning model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights.')
    
    args = parser.parse_args()
    evaluate(args.model_path) 