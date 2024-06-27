import os
import json
import torch
import pandas as pd
import argparse
from os.path import join
from models import LSTMModel, initialize_shared_embedding
from s2i import String2IntegerMapper
from metrics import get_perword_metrics_sentence
from myUtils import prepare_sequences, init_seed, prepare_tensors

def load_saved_model(path, model_class, device):
    """Load model from saved state."""
    # Load hyperparameters from JSON file
    hyperparams = json.load(open(join(path, "hyperparams.json")))

    # Initialize model parameters
    num_embeddings = hyperparams['output_size']
    embedding_dim = hyperparams['embedding_dim']

    # Initialize shared embedding layer
    shared_embedding = initialize_shared_embedding(num_embeddings, embedding_dim)

    # Create model instance
    model = model_class(hyperparams, device, shared_embedding)
    model.to(device)

    # Load saved model weights
    params_state_dict = torch.load(join(path, "model"), map_location=device)
    model.load_state_dict(params_state_dict, strict=False)

    return model

def preprocess_sentences(sentences, word_mappings):
    """Preprocess sentences by replacing out-of-vocabulary words with <UNK> and adding special tokens."""
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            if word in word_mappings.s2i:
                processed_sentence.append(word)
            else:
                processed_sentence.append('<UNK>')
        
        # Add start token
        processed_sentence.insert(0, '<s>')  
        # Add end token
        processed_sentence.append('</s>') 
        processed_sentences.append(processed_sentence)
    return processed_sentences

def sentences2indexs(sentences, word_mappings):
    """Convert sentences to indexes using the provided word mappings."""
    all_sentences_indexed = []
    for sentence in sentences:
        sentence_idxs = [word_mappings[word] for word in sentence]
        all_sentences_indexed.append(sentence_idxs)
    return all_sentences_indexed

def test_model(model, inputs_t, targets_t, sentences, output_file, device, language):
    """Test the model and save the results to a file."""
    header = ["actual_word", "correct", "previous_word", "predicted_word", "entropy", "entropy_top10", "surprisal", "target_in_top10", "lang", "perplexity_per_sentence"]
    perplexities = []

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Writing results to {output_file}")
    
    # Initialize a DataFrame to store all the results
    results_df = pd.DataFrame(columns=header)
    
    for i, (input_tensor, target_tensor) in enumerate(zip(inputs_t, targets_t)):
        # Generate model output
        output, _ = model(input_tensor.unsqueeze(0))
        output = output.squeeze(0)
        
        # Calculate perplexity for each sentence
        perplexity_per_sentence = torch.nn.NLLLoss()(output, target_tensor)
        perplexities.append(perplexity_per_sentence)
        
        # Get sentence metrics
        sentence = sentences[i]
        sentence_metrics = get_perword_metrics_sentence(output, target_tensor, sentence)
        sentence_metrics["lang"] = language
        sentence_metrics["perplexity_per_sentence"] = torch.exp(perplexity_per_sentence).item()
        
         # Collect each word prediction as a separate row
        for j in range(len(sentence_metrics["actual_word"])):
            row = {
                "actual_word": sentence_metrics["actual_word"][j],
                "correct": sentence_metrics["correct"][j],
                "previous_word": sentence_metrics["previous_word"][j],
                "predicted_word": sentence_metrics["predicted_word"][j],
                "entropy": sentence_metrics["entropy"][j],
                "entropy_top10": sentence_metrics["entropy_top10"][j],
                "surprisal": sentence_metrics["surprisal"][j],
                "target_in_top10": sentence_metrics["target_in_top10"][j],
                "lang": sentence_metrics["lang"],
                "perplexity_per_sentence": sentence_metrics["perplexity_per_sentence"]
            }
            results_df = results_df.append(row, ignore_index=True)
    
    # Save results to CSV
    results_df.to_csv(output_file, index=False)

    # Calculate and print average perplexity
    avg_perplexity = torch.exp(torch.stack(perplexities).mean()).item()
    print(f"Final perplexity for {model.__class__.__name__}: {avg_perplexity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed for training neural network.")
    parser.add_argument("--language", type=str, required=True, help="Language to train on.")
    parser.add_argument("--path_to_lstm_model", type=str, required=True, help="Directory containing saved LSTM model.")
    parser.add_argument("--input_data", type=str, required=True, help="File with sentences to test on.")
    args = parser.parse_args()

    # Initialize random seed for reproducibility
    init_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test sentences
    with open(args.input_data, 'r') as f:
        sentences = json.load(f)

    # Load LSTM model
    lstm_model = load_saved_model(args.path_to_lstm_model, LSTMModel, device)
    mappings = String2IntegerMapper.load(join(args.path_to_lstm_model, "w2i"))

    # Process sentences to replace out of vocabulary words with <UNK>
    processed_sentences = [[word if word in mappings.s2i else '<UNK>' for word in sent] for sent in sentences]

    # Add special tokens and print processed sentences
    processed_sentences = preprocess_sentences(processed_sentences, mappings)

    # Convert sentences to indexes and print indexed sentences
    all_sentences_indexed = sentences2indexs(processed_sentences, mappings)

    # Prepare sequences for the model
    inputs, targets = prepare_sequences(all_sentences_indexed)
    inputs_t, targets_t = prepare_tensors(inputs, targets, device)

    # Define output file for LSTM test results
    lstm_output_file = join(args.path_to_lstm_model, 'lstm_test_results.csv')

    # Test LSTM model and save results
    test_model(lstm_model, inputs_t, targets_t, processed_sentences, lstm_output_file, device, args.language)