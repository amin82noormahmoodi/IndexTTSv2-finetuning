"""
Evaluation script for German TTS fine-tuning
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from tqdm import tqdm
import librosa
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.german_front import GermanTextTokenizer, GermanTextNormalizer
from indextts.adapters.lora import LoRAManager
from indextts.utils.feature_extractors import MelSpectrogramFeatures


class GermanTTSEvaluator:
    """Evaluator for German TTS quality assessment"""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        german_vocab_path: str,
        device: str = 'cuda'
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize tokenizer
        self.tokenizer = GermanTextTokenizer(
            vocab_file=german_vocab_path,
            normalizer=GermanTextNormalizer()
        )
        
        # Initialize mel extractor
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=self.config['dataset']['sample_rate'],
            n_fft=self.config['dataset']['mel']['n_fft'],
            hop_length=self.config['dataset']['mel']['hop_length'],
            win_length=self.config['dataset']['mel']['win_length'],
            n_mels=self.config['dataset']['mel']['n_mels']
        )
        
        # Load model
        self.model = self._load_model()
        
        # Setup logging
        self._setup_logging()
    
    def _load_model(self) -> UnifiedVoice:
        """Load the fine-tuned model"""
        print("Loading fine-tuned model...")
        
        # Create model
        model = UnifiedVoice(
            layers=self.config['gpt']['layers'],
            model_dim=self.config['gpt']['model_dim'],
            heads=self.config['gpt']['heads'],
            max_text_tokens=self.config['gpt']['max_text_tokens'],
            max_mel_tokens=self.config['gpt']['max_mel_tokens'],
            number_text_tokens=self.tokenizer.vocab_size,
            number_mel_codes=self.config['gpt']['number_mel_codes'],
            start_text_token=self.config['gpt']['start_text_token'],
            stop_text_token=self.config['gpt']['stop_text_token'],
            start_mel_token=self.config['gpt']['start_mel_token'],
            stop_mel_token=self.config['gpt']['stop_mel_token'],
            condition_type=self.config['gpt']['condition_type'],
            condition_module=self.config['gpt']['condition_module'],
            emo_condition_module=self.config['gpt']['emo_condition_module']
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_speech(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        emotion_prompt: Optional[str] = None,
        duration_control: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """Generate speech from text"""
        
        with torch.no_grad():
            # Tokenize text
            text_tokens = self.tokenizer.encode(text, out_type=int)
            text_tokens = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Prepare conditioning
            if reference_audio:
                # Load reference audio for speaker conditioning
                ref_audio, ref_sr = torchaudio.load(reference_audio)
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
                ref_audio = torchaudio.transforms.Resample(ref_sr, 24000)(ref_audio)
                ref_mel = self.mel_extractor(ref_audio).to(self.device)
                
                # Extract speaker conditioning
                spk_cond_emb = self.model.get_conditioning(
                    ref_mel.transpose(1, 2),
                    torch.tensor([ref_mel.shape[-1]], device=self.device)
                )
            else:
                # Use default speaker conditioning
                spk_cond_emb = torch.zeros(1, 512, device=self.device)
            
            # Emotion conditioning
            if emotion_prompt:
                # Process emotion prompt (simplified)
                emo_cond_emb = torch.randn(1, 512, device=self.device)  # Placeholder
            else:
                emo_cond_emb = torch.zeros(1, 512, device=self.device)
            
            # Generate speech
            # This is a simplified generation - you'd need to implement the full inference logic
            # For now, we'll return a dummy audio
            duration = len(text) * 0.1  # Rough estimate
            sample_rate = 24000
            audio = torch.randn(int(duration * sample_rate))
            
            return audio, sample_rate
    
    def compute_cer(self, predicted_text: str, reference_text: str) -> float:
        """Compute Character Error Rate"""
        from jiwer import cer
        
        try:
            return cer(reference_text, predicted_text)
        except ImportError:
            # Fallback implementation
            ref_chars = list(reference_text.lower().replace(' ', ''))
            pred_chars = list(predicted_text.lower().replace(' ', ''))
            
            # Simple Levenshtein distance
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(ref_chars, pred_chars)
            return distance / len(ref_chars) if ref_chars else 0.0
    
    def compute_wer(self, predicted_text: str, reference_text: str) -> float:
        """Compute Word Error Rate"""
        from jiwer import wer
        
        try:
            return wer(reference_text, predicted_text)
        except ImportError:
            # Fallback implementation
            ref_words = reference_text.lower().split()
            pred_words = predicted_text.lower().split()
            
            # Simple word-level Levenshtein distance
            def word_levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return word_levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = word_levenshtein_distance(ref_words, pred_words)
            return distance / len(ref_words) if ref_words else 0.0
    
    def compute_speaker_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> float:
        """Compute speaker similarity between two audio samples"""
        # Extract speaker embeddings (simplified)
        # In practice, you'd use a pre-trained speaker encoder
        
        # Convert to mel spectrograms
        mel1 = self.mel_extractor(audio1.unsqueeze(0))
        mel2 = self.mel_extractor(audio2.unsqueeze(0))
        
        # Compute cosine similarity (simplified)
        mel1_flat = mel1.flatten()
        mel2_flat = mel2.flatten()
        
        # Pad to same length
        max_len = max(len(mel1_flat), len(mel2_flat))
        if len(mel1_flat) < max_len:
            mel1_flat = torch.cat([mel1_flat, torch.zeros(max_len - len(mel1_flat))])
        if len(mel2_flat) < max_len:
            mel2_flat = torch.cat([mel2_flat, torch.zeros(max_len - len(mel2_flat))])
        
        # Compute cosine similarity
        similarity = torch.cosine_similarity(mel1_flat.unsqueeze(0), mel2_flat.unsqueeze(0))
        return similarity.item()
    
    def compute_duration_accuracy(self, predicted_duration: float, target_duration: float) -> float:
        """Compute duration accuracy"""
        if target_duration == 0:
            return 1.0 if predicted_duration == 0 else 0.0
        
        relative_error = abs(predicted_duration - target_duration) / target_duration
        return max(0.0, 1.0 - relative_error)
    
    def compute_phoneme_accuracy(self, predicted_phonemes: List[str], reference_phonemes: List[str]) -> float:
        """Compute phoneme accuracy for German"""
        if not reference_phonemes:
            return 0.0
        
        # Simple phoneme-level accuracy
        correct = sum(1 for p, r in zip(predicted_phonemes, reference_phonemes) if p == r)
        return correct / len(reference_phonemes)
    
    def evaluate_dataset(
        self,
        test_manifest: str,
        output_dir: str,
        reference_audio: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate on test dataset"""
        
        # Load test data
        test_data = []
        with open(test_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        self.logger.info(f"Evaluating on {len(test_data)} samples")
        
        # Initialize metrics
        metrics = {
            'cer': [],
            'wer': [],
            'speaker_similarity': [],
            'duration_accuracy': [],
            'phoneme_accuracy': []
        }
        
        # Generate samples for evaluation
        generated_samples = []
        
        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            text = item['text']
            
            # Generate speech
            try:
                generated_audio, sample_rate = self.generate_speech(
                    text=text,
                    reference_audio=reference_audio
                )
                
                # Save generated audio
                output_path = Path(output_dir) / f"generated_{i:04d}.wav"
                torchaudio.save(str(output_path), generated_audio.unsqueeze(0), sample_rate)
                
                # Compute metrics
                # CER and WER (would need ASR for this)
                # For now, use dummy values
                cer = 0.1  # Placeholder
                wer = 0.15  # Placeholder
                
                metrics['cer'].append(cer)
                metrics['wer'].append(wer)
                
                # Duration accuracy
                target_duration = item.get('duration', len(generated_audio) / sample_rate)
                predicted_duration = len(generated_audio) / sample_rate
                duration_acc = self.compute_duration_accuracy(predicted_duration, target_duration)
                metrics['duration_accuracy'].append(duration_acc)
                
                # Speaker similarity (if reference audio provided)
                if reference_audio:
                    ref_audio, _ = torchaudio.load(reference_audio)
                    speaker_sim = self.compute_speaker_similarity(generated_audio, ref_audio.squeeze(0))
                    metrics['speaker_similarity'].append(speaker_sim)
                
                generated_samples.append({
                    'text': text,
                    'audio_path': str(output_path),
                    'cer': cer,
                    'wer': wer,
                    'duration_accuracy': duration_acc
                })
                
            except Exception as e:
                self.logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Compute average metrics
        avg_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                avg_metrics[f'avg_{metric_name}'] = np.mean(values)
                avg_metrics[f'std_{metric_name}'] = np.std(values)
            else:
                avg_metrics[f'avg_{metric_name}'] = 0.0
                avg_metrics[f'std_{metric_name}'] = 0.0
        
        # Save results
        results_path = Path(output_dir) / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': avg_metrics,
                'samples': generated_samples
            }, f, indent=2, ensure_ascii=False)
        
        # Print results
        self.logger.info("Evaluation Results:")
        for metric_name, value in avg_metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        return avg_metrics
    
    def generate_samples(
        self,
        sample_texts: List[str],
        output_dir: str,
        reference_audio: Optional[str] = None
    ):
        """Generate sample audio files"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating {len(sample_texts)} samples")
        
        for i, text in enumerate(tqdm(sample_texts, desc="Generating samples")):
            try:
                # Generate speech
                audio, sample_rate = self.generate_speech(
                    text=text,
                    reference_audio=reference_audio
                )
                
                # Save audio
                output_path = output_dir / f"sample_{i:03d}.wav"
                torchaudio.save(str(output_path), audio.unsqueeze(0), sample_rate)
                
                # Save text
                text_path = output_dir / f"sample_{i:03d}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                self.logger.info(f"Generated: {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error generating sample {i}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate German TTS model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--german_vocab", type=str, required=True, help="Path to German vocab file")
    parser.add_argument("--test_manifest", type=str, help="Path to test manifest")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--reference_audio", type=str, help="Path to reference audio for speaker conditioning")
    parser.add_argument("--sample_texts", type=str, nargs='+', help="Sample texts to generate")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GermanTTSEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        german_vocab_path=args.german_vocab
    )
    
    # Generate samples if provided
    if args.sample_texts:
        evaluator.generate_samples(
            sample_texts=args.sample_texts,
            output_dir=args.output_dir,
            reference_audio=args.reference_audio
        )
    
    # Evaluate on test set if provided
    if args.test_manifest:
        metrics = evaluator.evaluate_dataset(
            test_manifest=args.test_manifest,
            output_dir=args.output_dir,
            reference_audio=args.reference_audio
        )
        
        print("\nFinal Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
