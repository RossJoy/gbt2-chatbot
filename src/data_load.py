from tqdm import tqdm
from transformers import *
from data_process import *

import os
import json

def save_data(dialogues, name, dialogue_split_line, data_dir):  
    print(f"Saving {name} text file...")
    with open(f"{data_dir}/{name}.txt", 'w') as f:
        for dialogue in tqdm(dialogues):
            for utter in dialogue:
                f.write(f"{utter}\n")
            f.write(f"{dialogue_split_line}\n")
     
    print(f"Saving {name} idx file...")
    with open(f"{data_dir}/{name}.id", 'w') as f:
        for dialogue in tqdm(dialogues):
            for utter in dialogue:
                token_ids = tokenizer(utter)['input_ids']
                token_ids = ' '.join([str(idx) for idx in token_ids])
                f.write(f"{token_ids}\n")
            f.write(f"{dialogue_split_line}\n")

if __name__=='__main__':
    config_path = './config.json'
    
    print("Loading configurations...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('redrussianarmy/gpt2-turkish-cased')
    
    print("Loading & Merging the dataset...")
    train_dialogues, valid_dialogues, train_utter_num, valid_utter_num = load_persona_chat(tokenizer, config['train_frac'])
    
    if not os.path.isdir(config['data_dir']):
        os.mkdir(config['data_dir'])
    
    print("Saving train data...")
    save_data(train_dialogues, config['train_name'], config['dialogue_split_line'], config['data_dir'])
    print("Saving validation data...")
    save_data(valid_dialogues, config['valid_name'], config['dialogue_split_line'], config['data_dir'])            
    
    print("Data preprocess finished!")

    print(f"#################### Analysis on total data ####################")
    print(f"The number of train dialogues: {len(train_dialogues)}")
    print(f"The number of valid dialogues: {len(valid_dialogues)}")    
    print(f"The number of train utterances: {train_utter_num}")    
    print(f"The number of valid utterances: {valid_utter_num}")
    