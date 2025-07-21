import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import torch
from werkzeug.utils import secure_filename

from models.model import CNNtoRNN
from utils.data_loader import Vocabulary, load_captions_and_split
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build Vocabulary
print("Building vocabulary...")
captions_file = 'data/captions.txt'
captions_dict, train_keys, _, _ = load_captions_and_split(captions_file)
train_captions = [caption for key in train_keys for caption in captions_dict[key]]
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(train_captions)
print("Vocabulary built.")

# Model parameters
embed_size = 512
hidden_size = 512
num_layers = 2
vocab_size = len(vocab)
model_path = 'models/weights/image_captioning_final.pth' # CHANGE THIS IF YOU USE A DIFFERENT MODEL

# Load the trained model
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# --------------------

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    caption_tokens = model.caption_image(image_tensor, vocab)
    
    # Clean up tokens
    if caption_tokens and caption_tokens[0] == '<SOS>':
        caption_tokens = caption_tokens[1:]
    if caption_tokens and caption_tokens[-1] == '<EOS>':
        caption_tokens = caption_tokens[:-1]
        
    return ' '.join(caption_tokens)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            caption = generate_caption(filepath)
            
            return render_template('index.html', filename=filename, caption=caption)
            
    return render_template('index.html', filename=None, caption=None)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 