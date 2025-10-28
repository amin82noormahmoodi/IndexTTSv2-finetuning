"""
Data preprocessing pipeline for German TTS fine-tuning
"""
import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple
import librosa
import soundfile as sf

from indextts.utils.german_front import GermanTextNormalizer, create_german_sentencepiece_model
from indextts.utils.feature_extractors import MelSpectrogramFeatures


class GermanDataPreprocessor:
    """Preprocess German dataset for IndexTTS2 fine-tuning"""
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 sample_rate: int = 24000,
                 mel_channels: int = 100,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.mel_channels = mel_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "mel").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        
        # Initialize normalizer
        self.normalizer = GermanTextNormalizer()
        
        # Initialize mel spectrogram extractor
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=mel_channels
        )
    
    def load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata CSV file"""
        df = pd.read_csv(metadata_file, sep="|", header=None, names=["index", "audio_id", "text"])
        return df
    
    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Preprocess audio file"""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Normalize audio
        audio = audio / torch.max(torch.abs(audio))
        
        return audio.squeeze(0), self.sample_rate
    
    def extract_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from audio"""
        # Add batch dimension
        audio = audio.unsqueeze(0)
        
        # Extract mel spectrogram
        mel = self.mel_extractor(audio)
        
        return mel.squeeze(0)  # Remove batch dimension
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess German text"""
        # Apply German normalization
        normalized_text = self.normalizer.normalize(text)
        
        # Additional cleaning
        normalized_text = normalized_text.strip()
        
        return normalized_text
    
    def process_single_file(self, audio_id: str, text: str) -> Dict:
        """Process a single audio-text pair"""
        audio_path = self.data_dir / f"{audio_id}.wav"
        
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            return None
        
        try:
            # Preprocess audio
            audio, sr = self.preprocess_audio(str(audio_path))
            
            # Extract mel spectrogram
            mel = self.extract_mel_spectrogram(audio)
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                print(f"Warning: Empty text after preprocessing for {audio_id}")
                return None
            
            # Save processed files
            audio_output_path = self.output_dir / "audio" / f"{audio_id}.wav"
            mel_output_path = self.output_dir / "mel" / f"{audio_id}.npy"
            text_output_path = self.output_dir / "text" / f"{audio_id}.txt"
            
            # Save audio
            torchaudio.save(str(audio_output_path), audio.unsqueeze(0), sr)
            
            # Save mel spectrogram
            np.save(str(mel_output_path), mel.numpy())
            
            # Save text
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            return {
                "audio_id": audio_id,
                "audio_path": str(audio_output_path.relative_to(self.output_dir)),
                "mel_path": str(mel_output_path.relative_to(self.output_dir)),
                "text_path": str(text_output_path.relative_to(self.output_dir)),
                "text": processed_text,
                "duration": len(audio) / sr,
                "mel_frames": mel.shape[-1]
            }
            
        except Exception as e:
            print(f"Error processing {audio_id}: {e}")
            return None
    
    def process_dataset(self, metadata_file: str, max_files: int = None) -> List[Dict]:
        """Process entire dataset"""
        print("Loading metadata...")
        df = self.load_metadata(metadata_file)
        
        if max_files:
            df = df.head(max_files)
        
        print(f"Processing {len(df)} files...")
        
        processed_data = []
        failed_files = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            result = self.process_single_file(row['audio_id'], row['text'])
            
            if result is not None:
                processed_data.append(result)
            else:
                failed_files.append(row['audio_id'])
        
        print(f"Successfully processed {len(processed_data)} files")
        print(f"Failed to process {len(failed_files)} files")
        
        if failed_files:
            print("Failed files:", failed_files[:10])  # Show first 10
        
        return processed_data
    
    def create_training_manifest(self, processed_data: List[Dict], output_file: str):
        """Create training manifest file"""
        manifest_path = self.output_dir / output_file
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Training manifest saved to: {manifest_path}")
    
    def create_sentencepiece_model(self, processed_data: List[Dict], vocab_size: int = 16000):
        """Create German SentencePiece model from processed text"""
        # Extract all text
        all_texts = [item['text'] for item in processed_data]
        
        # Save texts to temporary file
        text_file = self.output_dir / "german_texts.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n')
        
        # Create SentencePiece model
        model_path = create_german_sentencepiece_model(
            text_file=str(text_file),
            vocab_size=vocab_size,
            model_prefix=str(self.output_dir / "german_bpe")
        )
        
        print(f"German SentencePiece model created: {model_path}")
        return model_path
    
    def create_statistics(self, processed_data: List[Dict]):
        """Create dataset statistics"""
        durations = [item['duration'] for item in processed_data]
        mel_frames = [item['mel_frames'] for item in processed_data]
        text_lengths = [len(item['text']) for item in processed_data]
        
        stats = {
            "total_files": len(processed_data),
            "total_duration_hours": sum(durations) / 3600,
            "avg_duration_seconds": np.mean(durations),
            "std_duration_seconds": np.std(durations),
            "min_duration_seconds": np.min(durations),
            "max_duration_seconds": np.max(durations),
            "avg_mel_frames": np.mean(mel_frames),
            "std_mel_frames": np.std(mel_frames),
            "avg_text_length": np.mean(text_lengths),
            "std_text_length": np.std(text_lengths)
        }
        
        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print("Dataset statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess German dataset for IndexTTS2")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to German dataset directory")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--vocab_size", type=int, default=16000, help="Vocabulary size for SentencePiece model")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = GermanDataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processed_data = preprocessor.process_dataset(
        metadata_file=args.metadata_file,
        max_files=args.max_files
    )
    
    # Create training manifest
    preprocessor.create_training_manifest(processed_data, "train_manifest.jsonl")
    
    # Create SentencePiece model
    preprocessor.create_sentencepiece_model(processed_data, args.vocab_size)
    
    # Create statistics
    preprocessor.create_statistics(processed_data)
    
    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()
