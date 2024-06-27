import json
import os

def preprocess_and_check_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    log_messages = []
    cleaned_data = []
    initial_sentence_count = len(data)

    # Check if data is a list
    if isinstance(data, list):
        log_messages.append("Data is a list.")
    else:
        log_messages.append("Error: Data is not a list.")

    # Remove empty lists and check items
    for i, item in enumerate(data):
        if not item:  # Empty list
            log_messages.append(f"Error: Empty list found at position {i}.")
            continue
        
        # Check if item is a list
        if not isinstance(item, list):
            log_messages.append(f"Error: Item {i} is not a list. It is of type {type(item)}.")
            continue

        # Check for extreme lengths
        if len(item) > 100:  # Threshold for sentences that are too long
            log_messages.append(f"Error: Item {i} is too long with {len(item)} words.")
            continue
        
        # Check if all words are strings and normalize
        new_item = []
        for word in item:
            if isinstance(word, str):
                # Split words with spaces
                if ' ' in word:
                    log_messages.append(f"Error: Item {i} contains a word with a space: '{word}'.")
                    new_item.extend(word.split())
                else:
                    new_item.append(word.lower())  # Normalize: lowercase
            else:
                log_messages.append(f"Error: Item {i} contains non-string elements: {word}.")
        
        cleaned_data.append(new_item)

    final_sentence_count = len(cleaned_data)
    log_messages.append(f"Initial number of sentences: {initial_sentence_count}")
    log_messages.append(f"Final number of sentences after cleaning: {final_sentence_count}")
    
    # Write log to file
    with open('data_check_log.txt', 'w') as log_file:
        for message in log_messages:
            log_file.write(message + "\n")
    
    # Write cleaned data to a new file
    with open('hi_checked_data.json', 'w') as cleaned_file:
        json.dump(cleaned_data, cleaned_file, ensure_ascii=False, indent=4)

# Path to the JSON file
file_path = os.path.join('hi_data.json')

# Use the function with your JSON file
preprocess_and_check_json_data(file_path)