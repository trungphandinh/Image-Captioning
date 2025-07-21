import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        "Greedy search to sample top suggestion"
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=20):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

    def beam_search(self, image_tensor, vocabulary, beam_size=3, max_length=20):
        device = image_tensor.device
        k = beam_size
        
        result_caption = []

        with torch.no_grad():
            feature = self.encoder(image_tensor)
            
            # The first input to the LSTM is the feature vector from the encoder
            hiddens, states = self.decoder.lstm(feature.unsqueeze(1), None)
            
            # Get the first set of top k words and their log probabilities
            output = self.decoder.linear(hiddens.squeeze(1))
            log_probs = F.log_softmax(output, dim=1)
            top_k_log_probs, top_k_words = log_probs.topk(k, 1)

            # Initialize the k beams
            live_beams = []
            for i in range(k):
                beam = {
                    'seq': [top_k_words[0][i].item()],
                    'states': states,
                    'log_prob': top_k_log_probs[0][i].item()
                }
                live_beams.append(beam)
            
            completed_beams = []
            
            for _ in range(max_length - 1):
                if not live_beams:
                    break
                
                # Get the last word and hidden states for all live beams for batch processing
                last_words = torch.tensor([beam['seq'][-1] for beam in live_beams]).to(device)
                
                h_state = torch.cat([beam['states'][0] for beam in live_beams], dim=1)
                c_state = torch.cat([beam['states'][1] for beam in live_beams], dim=1)
                
                # Predict next words
                embeddings = self.decoder.embed(last_words).unsqueeze(1)
                hiddens, new_states = self.decoder.lstm(embeddings, (h_state, c_state))
                output = self.decoder.linear(hiddens.squeeze(1))
                log_probs = F.log_softmax(output, dim=1)
                
                # Add current log probs to the scores of the live beams
                live_scores = torch.tensor([b['log_prob'] for b in live_beams]).unsqueeze(1).to(device)
                all_scores = log_probs + live_scores
                
                # Get top k scores from all k * vocab_size possibilities
                top_k_scores, top_k_indices = all_scores.view(-1).topk(k, 0, True, True)
                
                # Get the beam and word index for each top score
                prev_beam_indices = (top_k_indices / len(vocabulary)).long()
                next_word_indices = (top_k_indices % len(vocabulary)).long()
                
                # Create the new beams
                new_live_beams_temp = []
                for i in range(k):
                    prev_beam_idx = prev_beam_indices[i].item()
                    next_word_idx = next_word_indices[i].item()
                    
                    new_seq = live_beams[prev_beam_idx]['seq'] + [next_word_idx]
                    new_log_prob = top_k_scores[i].item()
                    
                    new_h = new_states[0][:, prev_beam_idx, :].unsqueeze(1)
                    new_c = new_states[1][:, prev_beam_idx, :].unsqueeze(1)
                    
                    new_beam = { 'seq': new_seq, 'states': (new_h, new_c), 'log_prob': new_log_prob }
                    
                    if next_word_idx == vocabulary.stoi['<EOS>']:
                        completed_beams.append(new_beam)
                    else:
                        new_live_beams_temp.append(new_beam)
                
                live_beams = new_live_beams_temp
                if len(completed_beams) >= k:
                    break
                    
            if not completed_beams:
                completed_beams = live_beams
                
            # Find the best beam (normalize by length)
            best_beam = sorted(completed_beams, key=lambda b: b['log_prob'] / len(b['seq']), reverse=True)[0]
            
            # Convert indices to words
            result_caption = [vocabulary.itos[idx] for idx in best_beam['seq']]

        return result_caption 

