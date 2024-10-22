import os
import pickle
from itertools import chain
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from chatbot_files.processing import Processing

class Dialogues(Processing):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        # self.dataset_list = ['daily_dialog', 'empathetic_dialogues', 'persona_chat', 'blended_skill_talk']
        self.dataset_list = ['daily_dialog']
        # self.dataset_list = ['daily_dialog', 'empathetic_dialogues', 'blended_skill_talk']
        super().__init__(tokenizer, args['train_frac'])

    def load(self):
        train_dataset = []
        valid_dataset = []

        # loading all datasets
        for dataset_name in self.dataset_list:
            print(f'Loading {dataset_name} dataset...')

            train_dialogues, valid_dialogues = self._load_dialog(dataset=dataset_name)
            train_dataset += train_dialogues
            valid_dataset += valid_dialogues
        
        return train_dataset, valid_dataset

    def save(self, prefix, tokenizer, dialogues):
        print(f'Saving {prefix} dialogues to file...')

        if not os.path.isdir(self.args["structure_dataset_dir"]):
            os.makedirs(self.args["structure_dataset_dir"])

        dialogues_path = f'{self.args["structure_dataset_dir"]}/{prefix}_dialogues.pickle'
        ids_path = f'{self.args["structure_dataset_dir"]}/{prefix}_ids.pickle'

        with open(dialogues_path, 'wb') as f:
            pickle.dump(dialogues, f)

        print(f'Saving {prefix} ids to file...')
        ids = []
        for dialogue in tqdm(dialogues):
            dialogue_ids = []
            for utter in dialogue:
                tokens = tokenizer.tokenize(utter)
                token_ids = tokenizer.encode(tokens)
                dialogue_ids.append(token_ids)
            ids.append(dialogue_ids)

        with open(ids_path, 'wb') as f:
            pickle.dump(ids, f)

        print('Saving complete!')

    def _load_dialog(self, dataset=None):
        if dataset == 'daily_dialog':
            return self._load_daily()
        elif dataset == 'empathetic_dialogues':
            return self._load_empathetic()
        elif dataset == 'persona_chat':
            return self._load_persona()
        elif dataset == 'blended_skill_talk':
            return self._load_blended()
        elif dataset == 'college_dataset':
            return self._load_college()

class DialoguesDataset(Dataset):
    def __init__(self, prefix, args):
        self.input_ids = []
        self.token_type_ids = []
        self.labels = []
        self._prepare_data(prefix, args)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]

    def _prepare_data(self, prefix, args):
        with open(f'{args["structure_dataset_dir"]}/{prefix}_ids.pickle', 'rb') as f: # trin_ids.pickle and valid_ids.pickle
            dials = pickle.load(f)

        for dial in tqdm(dials):
            hists = []
            for i, sentence in enumerate(dial):
                if i % 2 == 0:
                    hists.append([args['sp1_id']] + sentence)
                else:
                    hists.append([args['sp2_id']] + sentence)

            for i in range(len(hists)):
                if hists[i][0] == args['sp2_id']:
                    for j in range(0, i):
                        contexts = hists[j:i + 1]
                        if len(contexts) > args['max_history']:
                            num_exceeded = len(contexts) - args['max_history']
                            contexts = contexts[num_exceeded:]
                        if len(contexts) < 2:
                            break

                        input_ids = [args['bos_id']] + list(chain.from_iterable(contexts)) + [args['eos_id']]
                        if len(input_ids) <= args['max_len']:
                            start_sp_id, next_sp_id = contexts[0][0], contexts[1][0]
                            token_type_ids = [[start_sp_id] * len(ctx) if c % 2 == 0 else [next_sp_id] * len(ctx) for c, ctx in enumerate(contexts)]
                            token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [args['sp2_id']]

                            labels = [[-100] * len(ctx) if c < len(contexts) - 1 else [-100] + ctx[1:] for c, ctx in enumerate(contexts)]
                            labels = [-100] + list(chain.from_iterable(labels)) + [args['eos_id']]

                            self.input_ids.append(input_ids)
                            self.token_type_ids.append(token_type_ids)
                            self.labels.append(labels)

                            break

        del dials

# For Curpos Training

class Corpus(Processing):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.dataset_list = ['college_corpus']
        super().__init__(tokenizer, args['train_frac'])

    def load(self):
        # loading all datasets
        train_dataset, valid_dataset = self._college_corpus() ## Have to Modify
        return train_dataset, valid_dataset

    def save(self, prefix, tokenizer, lines):
        print(f'Saving {prefix} Corpus to file...')
        if not os.path.isdir(self.args["corpus_dataset_dir"]):
            os.makedirs(self.args["corpus_dataset_dir"])

        lines_path = f'{self.args["corpus_dataset_dir"]}/{prefix}_corpus.pickle'
        ids_path = f'{self.args["corpus_dataset_dir"]}/{prefix}_ids.pickle'
        
        with open(lines_path, 'wb') as f:
            pickle.dump(lines, f)   
        
        print(f'Saving {prefix} ids to file...')
        ids = [] 
        for utter in tqdm(lines):
            tokens = tokenizer.tokenize(utter)
            token_ids = tokenizer.encode(tokens)
            ids.append(token_ids)

        with open(ids_path, 'wb') as f:
            pickle.dump(ids, f)

        print('Saving complete!')

class CorpusDataSet(Dataset):
    def __init__(self, prefix, args):
        self.input_ids = []
        self.labels = []
        self.pad_token_id = args.get('eos_id', 0)  # Default padding token ID args['eos_id']
        self.max_length = args['max_len']  # Maximum length for truncation/padding
        # self.max_length = 768
        
        # Load the tokenized data from pickle and prepare input_ids and labels
        self._prepare_data(prefix, args)
    
    def _prepare_data(self, prefix, args):
        # Load pre-tokenized data from the pickle file
        with open(f'{args["corpus_dataset_dir"]}/{prefix}_ids.pickle', 'rb') as f:
            dials = pickle.load(f)  # Load the tokenized data
        
        # Process each dialogue (or sequence of token IDs)
        for token_ids in dials:
            # Ensure each sequence is truncated/padded to max_length
            input_ids = token_ids[:self.max_length]  # Truncate to max_length
            input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))  # Pad to max_length

            # Create labels by shifting the input_ids
            labels = input_ids[1:] + [self.pad_token_id]  # Shift input_ids to create labels

            # Store input_ids and labels directly as lists
            self.input_ids.append(input_ids)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return input_ids and labels as lists
        return self.input_ids[idx], self.labels[idx]
