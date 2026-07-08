import json 
from tqdm import tqdm 
from collections import defaultdict
from typing import List, Tuple, Optional



class TigrinyaBPE:

    """
    Character Based Byte Pair Encoding tokenizer for Tigrinya language with custom vocabulary size and special tokens.

    Features:
    - Train custom BPE tokenizer 
    - Encode and decode text 
    - Save  and load trained tokenizer 
    """

    def __init__(self, vocab_size: int = 8000, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]']
        self.vocab = set()
        self.merges = []
        self.word_freqs = defaultdict(int)
        self.is_trained = False 
    

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for training (corpus specific)"""
        import re 

        # Basic cleaning 
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n', ' ', text)

        return text 

    def _get_word_tokens(self, corpus: str) -> List[List[str]]:
        """Convert corpus to list of character-tokenized words."""
        words = corpus.split()
        return [list(word) for word in words]

    
    def _count_pairs(self, words_list: List[List[str]]) -> Optional[Tuple[str, str]]:
        """Count adjacent character pairs and return the most frequent one."""

        pairs = defaultdict(int)

        for word in words_list:
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pairs[pair] += 1 
        
        if not pairs:
            return None 
        
        best_pair = max(pairs, key=pairs.get)

        return best_pair
    

    def _merge_pairs(self, words_list: List[List[str]], best_pair: Tuple[str, str]) -> List[List[str]]:
        """Merge the most frequent pair in all words."""

        new_words_list = []

        for word in words_list:

            i=0
            new_word = []
            while i < len(word):

                if (i < len(word)-1 and 
                    word[i] == best_pair[0] and 
                    word[i+1] == best_pair[1]):

                    new_word.append(''.join(best_pair))
                    i+=2
                
                else:
                    new_word.append(word[i])
                    i+=1

            new_words_list.append(new_word)
        
        return new_words_list

    
    def train(self, corpus: str, verbose: bool = True) -> None:

        """
        Train the BPE tokenizer on the given corpus.
        
        Args:
            corpus: Training text corpus
            verbose: Whether to show progress bar
        """

        print("Starting BPE training...")

        corpus = self._preprocess_text(corpus)

        words_list = self._get_word_tokens(corpus)

        self.vocab = set()
        for word in words_list:
            self.vocab.update(word)
        
        self.vocab.update(self.special_tokens)

        if verbose:
            pbar = tqdm(desc="Tigrinya BPE Training", total = self.vocab_size - len(self.vocab))
        
        while len(self.vocab) < self.vocab_size:
            best_pair = self._count_pairs(words_list)

            if not best_pair:
                print("No more pairs to merge. Training complete")
                break 

            words_list = self._merge_pairs(words_list, best_pair)
            self.vocab.add(''.join(best_pair))
            self.merges.append(best_pair)

            if verbose:
                pbar.update(1)
                pbar.set_postfix({
                    "vocab_size": len(self.vocab),
                    "best_pair": f"{best_pair[0]}{best_pair[1]}"
                })
    
        if verbose:
            pbar.close()

        self.is_trained = True 
        print(f'BPE training complete! Final vocabulary size: {len(self.vocab)}')
    

    def encode(self, text: str) -> List[str]:

        """
        Encode text into BPE tokens.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of BPE tokens
        """

        if not self.is_trained:
            raise ValueError("The tokenizer is not trained. Please call train() first")
        
        if not text.strip():
            return []

        text = self._preprocess_text(text)
        
        words = text.split()

        tokens = []

        for i, word in enumerate(words):

            word_tokens = self._apply_bpe_to_word(word)

            if word_tokens:
                tokens.extend(word_tokens)
            
            if i < len(words)-1:
                tokens.append('Ġ')

        return tokens 
    

    def _apply_bpe_to_word(self, word):
        """Apply BPE to a single word using the learned merge operations."""

        word_tokens = list(word)

        for merge in self.merges:

            i = 0
            new_tokens = []

            while i < len(word_tokens):

                if (i < len(word_tokens)-1 and 
                    word_tokens[i] == merge[0] and 
                    word_tokens[i+1] == merge[1]):

                    new_tokens.append(''.join(merge))
                    i+=2
                else:
                    new_tokens.append(word_tokens[i])
                    i+=1
            
            word_tokens = new_tokens
        
        return word_tokens
    
    
    def decode(self, tokens: str) -> str:
        """
        Decode BPE tokens back to text.
        
        Args:
            tokens: List of BPE tokens
            
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        
        if not tokens:
            return ''
        
        text = ''.join(tokens).replace('Ġ', ' ')

        return text


    def save(self, filepath: str) -> None:

        """
        Save trained tokenizer to file 

        Args:
            filepath: Path to save the tokenizer 
        """

        if not self.is_trained:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'vocab': list(self.vocab),
            'merges': self.merges,
            'is_trained': self.is_trained
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f'Tokenizer saved to {filepath}')

    
    def load(self, filepath: str) -> None:

        """
        Load a trained tokenizer from file 

        Args:
            filepath: the path the tokenizer is saved
        """

        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        self.vocab_size = tokenizer_data['vocab_size']
        self.special_tokens = tokenizer_data['special_tokens']
        self.vocab = set(tokenizer_data['vocab'])
        self.merges = tokenizer_data['merges']
        self.is_trained = tokenizer_data['is_trained']

        print(f'Tokenizer loaded from {filepath}')

        return self
    
    def __str__(self) -> str:
        """ String representation of the tokenizer """
        status = "trained" if self.is_trained else "Not trained"
        return f'TigrinyaBPE(vocab_size={self.vocab_size}), status={status}'
    
    def __repr__(self) -> str:
        """String Representation """
        return self.__str__()

            





        








    


     
