"""
German-specific text processing for IndexTTS2
"""
import re
import unicodedata
from typing import List, Dict, Optional
import sentencepiece as spm
from .front import TextTokenizer


class GermanTextNormalizer:
    """German text normalizer with language-specific rules"""
    
    def __init__(self):
        self.german_contractions = {
            "am": "an dem",
            "ans": "an das", 
            "aufs": "auf das",
            "beim": "bei dem",
            "durchs": "durch das",
            "fürs": "für das",
            "hinterm": "hinter dem",
            "hinters": "hinter das",
            "ins": "in das",
            "im": "in dem",
            "überm": "über dem",
            "übers": "über das",
            "ums": "um das",
            "unterm": "unter dem",
            "unters": "unter das",
            "vom": "von dem",
            "vors": "vor das",
            "vorm": "vor dem",
            "zur": "zu der",
            "zum": "zu dem",
            "zurückm": "zurück dem",
            "zurücks": "zurück das"
        }
        
        # German number words
        self.number_words = {
            "null": "0", "eins": "1", "zwei": "2", "drei": "3", "vier": "4",
            "fünf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9",
            "zehn": "10", "elf": "11", "zwölf": "12", "dreizehn": "13",
            "vierzehn": "14", "fünfzehn": "15", "sechzehn": "16",
            "siebzehn": "17", "achtzehn": "18", "neunzehn": "19",
            "zwanzig": "20", "dreißig": "30", "vierzig": "40",
            "fünfzig": "50", "sechzig": "60", "siebzig": "70",
            "achtzig": "80", "neunzig": "90", "hundert": "100",
            "tausend": "1000", "million": "1000000"
        }
        
        # German ordinal numbers
        self.ordinal_words = {
            "erste": "1.", "zweite": "2.", "dritte": "3.", "vierte": "4.",
            "fünfte": "5.", "sechste": "6.", "siebte": "7.", "achte": "8.",
            "neunte": "9.", "zehnte": "10."
        }
        
        # Common German abbreviations
        self.abbreviations = {
            "z.B.": "zum Beispiel",
            "bzw.": "beziehungsweise", 
            "usw.": "und so weiter",
            "etc.": "et cetera",
            "ca.": "circa",
            "d.h.": "das heißt",
            "Prof.": "Professor",
            "Dr.": "Doktor",
            "etc.": "et cetera"
        }
    
    def normalize_numbers(self, text: str) -> str:
        """Convert German number words to digits"""
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.number_words:
                result.append(self.number_words[word_lower])
            elif word_lower in self.ordinal_words:
                result.append(self.ordinal_words[word_lower])
            else:
                result.append(word)
        
        return " ".join(result)
    
    def expand_contractions(self, text: str) -> str:
        """Expand German contractions"""
        words = text.split()
        result = []
        
        for word in words:
            if word.lower() in self.german_contractions:
                result.append(self.german_contractions[word.lower()])
            else:
                result.append(word)
        
        return " ".join(result)
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common German abbreviations"""
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize German punctuation"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Normalize quotes
        text = re.sub(r'[""„"]', '"', text)
        text = re.sub(r'[""‚"]', "'", text)
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        return text.strip()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        # Normalize to NFD form
        text = unicodedata.normalize('NFD', text)
        return text
    
    def normalize(self, text: str) -> str:
        """Main normalization pipeline for German text"""
        if not text or not text.strip():
            return ""
        
        # Unicode normalization
        text = self.normalize_unicode(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # Normalize numbers
        text = self.normalize_numbers(text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        return text


class GermanTextTokenizer(TextTokenizer):
    """German-specific text tokenizer extending the base TextTokenizer"""
    
    def __init__(self, vocab_file: str, normalizer: Optional[GermanTextNormalizer] = None):
        super().__init__(vocab_file, normalizer)
        if normalizer is None:
            self.normalizer = GermanTextNormalizer()
    
    def preprocess_german_text(self, text: str) -> str:
        """German-specific text preprocessing"""
        # Handle German-specific characters and diacritics
        text = text.replace('ß', 'ss')  # Eszett normalization
        
        # Handle umlauts - keep them for proper pronunciation
        # text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Override tokenize to include German preprocessing"""
        if not text or not text.strip():
            return []
        
        # Apply German-specific preprocessing
        text = self.preprocess_german_text(text)
        
        # Apply normalization
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        # Apply pre-tokenizers
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        
        # Tokenize with SentencePiece
        return self.sp_model.Encode(text, out_type=str)
    
    def encode(self, text: str, **kwargs):
        """Override encode to include German preprocessing"""
        if not text or not text.strip():
            return []
        
        # Apply German-specific preprocessing
        text = self.preprocess_german_text(text)
        
        # Apply normalization
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        # Apply pre-tokenizers
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)


def create_german_sentencepiece_model(
    text_file: str,
    vocab_size: int = 16000,
    model_prefix: str = "german_bpe",
    character_coverage: float = 0.9995
) -> str:
    """
    Create a German SentencePiece model from text data
    
    Args:
        text_file: Path to text file with German sentences
        vocab_size: Vocabulary size for the model
        model_prefix: Prefix for output model files
        character_coverage: Character coverage for the model
    
    Returns:
        Path to the created model file
    """
    import sentencepiece as spm
    
    # SentencePiece training parameters optimized for German
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type='bpe',
        max_sentence_length=4192,
        shuffle_input_sentence=True,
        input_sentence_size=1000000,
        seed_sentencepiece_size=1000000,
        shrinking_factor=0.75,
        num_threads=16,
        num_sub_iterations=2,
        max_sentencepiece_length=16,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=False,
        split_digits=True,
        vocabulary_output_piece_score=True,
        hard_vocab_limit=True,
        use_all_vocab=False,
        byte_fallback=True,
        vocabulary_threshold=50,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        pad_piece='<pad>',
        unk_surface='<unk>',
        train_extremely_large_corpus=False,
        enable_differential_privacy=False,
        differential_privacy_noise_level=0.0,
        differential_privacy_clipping_threshold=0.0,
        user_defined_symbols=['<speaker>', '<emotion>', '<duration>']
    )
    
    return f"{model_prefix}.model"


if __name__ == "__main__":
    # Example usage
    normalizer = GermanTextNormalizer()
    test_text = "Das ist ein Test mit Umlauten: ä, ö, ü und ß. Z.B. ca. 20% der Wörter."
    normalized = normalizer.normalize(test_text)
    print(f"Original: {test_text}")
    print(f"Normalized: {normalized}")
