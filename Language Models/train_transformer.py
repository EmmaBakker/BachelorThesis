import argparse
import os
import sys
import time
import torch
import random
import numpy as np
import json
import csv
from models import NWPTransformer, initialize_shared_embedding
from myUtils import prepare_tensors, input_to_indexs, init_seed, prepare_sequences, create_batches, rare_words_to_unknown, sentences2indexs
from convergence import ConvergenceCriterion, ConvergenceCriteria
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Log CUDA usage
def log_cuda_usage(fh, device):
    if device.type == 'cuda':
        fh.write("Using CUDA: {}\n".format(torch.cuda.get_device_name(device)))
        fh.write('Memory Usage:\n')
        fh.write('Allocated: %f GB\n' % (torch.cuda.memory_allocated(device)/1024**3))
        fh.write('Cached:   %f GB\n' % (torch.cuda.memory_reserved(device)/1024**3))

# Calculate perplexity from loss
def calculate_perplexity(loss):
    try:
        return np.exp(loss)
    except OverflowError:
        return float('inf')

# Count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_convergence_criteria():
    return ConvergenceCriteria([
        ConvergenceCriterion("loss", {"max_epochs_stagnated": 10, "min_change": 0.0025}),
        ConvergenceCriterion("perplexity", {"max_epochs_stagnated": 10, "min_change": 0.0025})
    ], "or")

# Train and tune the Transformer models
def train_and_tune_transformer(config, batched_inputs_t, batched_targets_t, batched_inputs_valid_t, batched_targets_valid_t, device, flog, shared_embedding, word_mappings, args, model_idx):
    # Initialize the model, optimizer and loss function
    model = NWPTransformer(config, device, shared_embedding)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    convergence_criteria = create_convergence_criteria()

    # Move model to the device
    model.to(device)
    loss_function = torch.nn.NLLLoss(ignore_index=0)

    loss_train_all_epochs = []
    loss_valid_all_epochs = []
    perplexity_all_epochs = []
    converged = False
    epoch = 0

    while not converged and epoch < 50:  # Train for up to 50 epochs
        epoch += 1
        start_time = time.perf_counter()
        epoch_train_losses = []
        epoch_valid_losses = []

        torch.cuda.empty_cache()

        # Training loop
        model.train()
        for input_tensor, target_tensor in zip(batched_inputs_t, batched_targets_t):
            optimizer.zero_grad()
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            output, _ = model(input_tensor)
            output_flat = output.view(-1, output.shape[-1])
            target_flat = target_tensor.view(-1)
            loss = loss_function(output_flat, target_flat)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_losses.append(loss.item())

        # Validation loop
        model.eval()
        with torch.no_grad():
            for input_tensor, target_tensor in zip(batched_inputs_valid_t, batched_targets_valid_t):
                input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
                output, _ = model(input_tensor)
                output_flat = output.view(-1, output.shape[-1])
                target_flat = target_tensor.view(-1)
                loss = loss_function(output_flat, target_flat)
                epoch_valid_losses.append(loss.item())

        # Calculate average losses and perplexity
        average_train_loss = np.mean(epoch_train_losses)
        average_valid_loss = np.mean(epoch_valid_losses)
        train_perplexity = calculate_perplexity(average_train_loss)
        valid_perplexity = calculate_perplexity(average_valid_loss)
        loss_train_all_epochs.append(average_train_loss)
        loss_valid_all_epochs.append(average_valid_loss)
        perplexity_all_epochs.append(valid_perplexity)

        # Check convergence criteria
        converged = convergence_criteria.update_state(epoch, loss_train_all_epochs, None, None, None, perplexity_all_epochs).converged()
        
        # Scheduler step
        scheduler.step(average_valid_loss)

        # Log training progress
        with open(flog, "a") as fh:
            log_cuda_usage(fh, device)
            fh.write(f"Epoch: {epoch} - Train Loss: {average_train_loss:.4f} - Train Perplexity: {train_perplexity:.4f} - Valid Loss: {average_valid_loss:.4f} - Valid Perplexity: {valid_perplexity:.4f} - Time: {time.perf_counter() - start_time:.2f}s\n")
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} - Train Loss: {average_train_loss:.4f} - Train Perplexity: {train_perplexity:.4f} - Valid Loss: {average_valid_loss:.4f} - Valid Perplexity: {valid_perplexity:.4f} - Time: {time.perf_counter() - start_time:.2f}s\n")

        # Save intermediate models every 5 epochs
        if epoch % 5 == 0:
            model_save_path = os.path.join(args.results_dir, f'Tf_{args.language}_MODEL{model_idx}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model.save_model(model_save_path, word_mappings, epoch, args, 1, additional_description=f"transformer_intermediate")

    return average_train_loss, average_valid_loss, train_perplexity, valid_perplexity, epoch, config['lr'], model


def process_data(sentences, min_frequency, word_mappings=None, is_training=True):
    beginning = "<s>"
    ending = "</s>"
    
    if is_training:
        word_mappings, all_sentences_indexed = input_to_indexs(sentences, min_frequency)
    else:
        # Set out-of-vocabulary words to unknown token
        processed_sentences = [[word if word in word_mappings.s2i else '<UNK>' for word in sent] for sent in sentences]

        # Handle rare words for Wiki data, not for data accompanying eye-tracking measures
        processed_sentences = rare_words_to_unknown(processed_sentences, min_frequency)

        # Convert sentences to indexes
        all_sentences_indexed = sentences2indexs(processed_sentences, beginning, ending, word_mappings)

    inputs, targets = prepare_sequences(all_sentences_indexed)
    return inputs, targets, word_mappings


def main(args):
    try:
        # Initialize seed for reproducibility
        init_seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create results directory
        args.results_dir = f"{args.language}_Tf_Results"
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

        # Load and preprocess data
        with open(args.train_data, 'r') as f:
            train_set = json.load(f)
        with open(args.validation_data, 'r') as f:
            validation_set = json.load(f)

        # Prepare training data
        inputs_train, targets_train, word_mappings = process_data(train_set, args.min_frequency, is_training=True)
        batched_inputs_train, batched_targets_train = create_batches(inputs_train, targets_train, args.batch_size)
        batched_inputs_t, batched_targets_t = prepare_tensors(batched_inputs_train, batched_targets_train, device)

        # Prepare validation data
        inputs_valid, targets_valid, _ = process_data(validation_set, args.min_frequency, word_mappings, is_training=False)
        batched_inputs_valid, batched_targets_valid = create_batches(inputs_valid, targets_valid, args.batch_size)
        batched_inputs_valid_t, batched_targets_valid_t = prepare_tensors(batched_inputs_valid, batched_targets_valid, device)

        flog = os.path.join(args.results_dir, f"Training_log_Tf_{args.language}_seed_{args.seed}.txt")
        with open(flog, "w") as fh:
            fh.write(f"Training on file {args.train_data}\n")
            log_cuda_usage(fh, device)

        num_embeddings = len(word_mappings.s2i)
        shared_embedding = initialize_shared_embedding(num_embeddings, 400).to(device)

        # Define the hyperparameter ranges
        fc_sizes = [512, 1024]
        n_layers = [1, 2]
        h = [4, 8] 
        learning_rates = [0.001, 0.005, 0.0001]

        # Generate the grid of hyperparameters
        grid_params = []
        for fc_size in fc_sizes:
            for num_layers in n_layers:
                for num_heads in h:
                    for lr in learning_rates:
                        grid_params.append({
                            'embed': {
                                'n_embeddings': num_embeddings,
                                'embedding_dim': 400,
                                'sparse': False,
                                'padding_idx': 0
                            },
                            'tf': {
                                'in_size': 400,
                                'fc_size': fc_size,
                                'n_layers': num_layers,
                                'h': num_heads,
                                'max_len': 102
                            },
                            'cuda': device.type == 'cuda',
                            'lr': lr
                        })

        # Train and evaluate Transformer models
        transformer_results = []
        losses_train = []
        losses_valid = []
        epochs_list = []
        weights_list = []

        for index, config in enumerate(grid_params):
            print(f"One setting: {config}")
            model_instance = NWPTransformer(config, device, shared_embedding)
            print(f"Number of parameters: {count_parameters(model_instance)}")

            result = train_and_tune_transformer(config, batched_inputs_t, batched_targets_t, batched_inputs_valid_t, batched_targets_valid_t, device, flog, shared_embedding, word_mappings, args, index)
            transformer_results.append(result)

            average_train_loss, average_valid_loss, _, _, epoch, lr, model = result
            losses_train.append(average_train_loss)
            losses_valid.append(average_valid_loss)
            epochs_list.append(epoch)
            weights_list.append(count_parameters(model))
            with open(os.path.join(args.results_dir, f"hyperparameter_values_{args.language}_{args.seed}_min_freq_{args.min_frequency}.txt"), "a") as f:
                f.write(f"Train loss: {average_train_loss}   Validation loss: {average_valid_loss}   Hyperparameters: {config}   Epochs: {epoch}\n")

        # Save results to CSV
        with open(os.path.join(args.results_dir, f"{args.language}_hyperparameter_tuning_results_transformer.csv"), "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model Type", "Hyperparameters", "Train Loss", "Valid Loss", "Train Perplexity", "Valid Perplexity", "Epochs", "Learning Rate", "Weights"])
            for config, train_loss, valid_loss, epoch, weights, model in zip(grid_params, losses_train, losses_valid, epochs_list, weights_list, transformer_results):
                model = model[-1]
                writer.writerow(['Transformer', config, train_loss, valid_loss, calculate_perplexity(train_loss), calculate_perplexity(valid_loss), epoch, lr, weights])
                model_save_path = os.path.join(args.results_dir, f'Tf_{args.language}_MODEL{grid_params.index(config)}')
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                model.save_model(model_save_path, word_mappings, epoch, args, grid_params.index(config), additional_description="transformer")

        # Find the best hyperparameters based on validation perplexity
        best_transformer_valid_loss = min(losses_valid)
        best_transformer_index = losses_valid.index(best_transformer_valid_loss)
        best_transformer_config = grid_params[best_transformer_index]
        best_model = transformer_results[best_transformer_index][-1]

        print(f"Best Transformer Config: {best_transformer_config}, Validation Perplexity: {calculate_perplexity(best_transformer_valid_loss)}, Weights: {count_parameters(best_model)}")

    except Exception as e:
        print(f"There is an error {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--language", type=str, required=True, help="Language to train on.")
    parser.add_argument("--min_frequency", type=int, default=24, help="Words with a frequency smaller than this are set to unknown.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of sentences per batch.")
    parser.add_argument("--train_data", type=str, required=True, help="File with the training set.")
    parser.add_argument("--validation_data", type=str, required=True, help="File with the validation set.")
    args = parser.parse_args()
    main(args)

