from tqdm import tqdm
from datasets import *


# For all
space = 'Ġ'
pre_quote = '’'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']

# For empathetic dialogues
exclude_symbol = "_conv"
comma_symbol = "_comma_"

# For persona chat
# persona_chat_url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
persona_chat_url = "../dataset/tpc.json"
silence_symbol = "__ SILENCE __"

def load_persona_chat(tokenizer, train_frac):
    import json
    with open(persona_chat_url) as f:
        dataset = json.loads(f.read())
        
    train_data = dataset['train']
    valid_data = dataset['valid']
    total_data = train_data + valid_data
    total_dialogues = []
    
    for obj in tqdm(total_data):
        dialogue = obj['ifadeler'][-1]['tarih']
        new_dialogue = []
        
        for i, utter in enumerate(dialogue):
            if utter.strip() != silence_symbol:
                token_list = tokenizer.tokenize(utter.strip())
                new_token_list = process_token_list(token_list)
                text = tokenizer.convert_tokens_to_string(new_token_list)
                new_dialogue.append(text)
        
        total_dialogues.append(new_dialogue)
        
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)] # remaining
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):] # first %15
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num

def process_token_list(token_list):
    token_list[0] = token_list[0].capitalize()
    
    quote_count = 0
    for i, token in enumerate(token_list):
        if space in token:
            if token[1:] in end_marks or token[1:] in abbreviations:
                token_list[i] = token[1:]
                
            if token[1:] == quotes[1]:
                if i<len(token_list)-1:
                    if token_list[i+1] in abbreviations or (token_list[i+1][0] == space and token_list[i+1][1:] in abbreviations):
                        token_list[i] = token[1:]
                        
        if token[0] == space and token[1:] in quotes:
            if quote_count % 2 == 1:
                token_list[i] = token[1:]
                quote_count = 0
            else:
                if i<len(token_list)-1 and token_list[i+1][0] == space:
                    token_list[i+1] = token_list[i+1][1:]
                quote_count += 1
                
        if token in end_marks or token[1:] in end_marks:
            if i<len(token_list)-1:
                if token_list[i+1][0] != space:
                    token_list[i+1] = space + token_list[i+1].capitalize()
                else:
                    token_list[i+1] = space + token_list[i+1][1:].capitalize()
                
    new_token_list = [token for token in token_list if token != space and len(token)>0]
    if new_token_list[-1] not in end_marks:
        new_token_list.append(end_marks[0])
        
    return new_token_list
