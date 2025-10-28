"""
German fine-tuning script for IndexTTS2
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
import wandb
from transformers import get_linear_schedule_with_warmup

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.german_front import GermanTextTokenizer, GermanTextNormalizer
from indextts.adapters.lora import add_lora_to_model, LoRAManager
from indextts.utils.feature_extractors import MelSpectrogramFeatures


class GermanTTSDataset(Dataset):
    """Dataset for German TTS fine-tuning"""
    
    def __init__(
        self,
        manifest_file: str,
        tokenizer: GermanTextTokenizer,
        mel_extractor: MelSpectrogramFeatures,
        max_text_length: int = 600,
        max_mel_length: int = 1815
    ):
        self.manifest_file = manifest_file
        self.tokenizer = tokenizer
        self.mel_extractor = mel_extractor
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        
        # Load manifest
        self.data = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio and extract mel spectrogram
        audio_path = item['audio_path']
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono
        
        # Extract mel spectrogram
        mel = self.mel_extractor(audio)
        mel = mel.squeeze(0)  # Remove batch dimension
        
        # Tokenize text
        text = item['text']
        text_tokens = self.tokenizer.encode(text, out_type=int)
        
        # Truncate if too long
        if len(text_tokens) > self.max_text_length:
            text_tokens = text_tokens[:self.max_text_length]
        
        if mel.shape[-1] > self.max_mel_length:
            mel = mel[:, :self.max_mel_length]
        
        return {
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'mel_spectrogram': mel,
            'text': text,
            'audio_id': item['audio_id']
        }


class GermanTTSFineTuner:
    """Fine-tuner for German TTS adaptation"""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        output_dir: str,
        german_vocab_path: str,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        device: str = 'cuda'
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
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
        
        # Setup LoRA if enabled
        if self.use_lora:
            self.lora_manager = add_lora_to_model(
                model=self.model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                target_modules=[
                    'gpt.h.*.attn.c_attn',
                    'gpt.h.*.attn.c_proj',
                    'gpt.h.*.mlp.c_fc',
                    'gpt.h.*.mlp.c_proj'
                ]
            )
        else:
            self.lora_manager = None
        
        # Setup logging
        self._setup_logging()
    
    def _load_model(self) -> UnifiedVoice:
        """Load the pre-trained model"""
        print("Loading pre-trained model...")
        
        # Create model with German vocabulary size
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
        
        # Load pre-trained weights
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Handle vocabulary size mismatch
        if 'text_embedding.weight' in checkpoint:
            old_vocab_size = checkpoint['text_embedding.weight'].shape[0]
            new_vocab_size = self.tokenizer.vocab_size
            
            if old_vocab_size != new_vocab_size:
                print(f"Resizing text embedding from {old_vocab_size} to {new_vocab_size}")
                
                # Create new embedding layer
                old_embedding = checkpoint['text_embedding.weight']
                new_embedding = torch.randn(new_vocab_size, old_embedding.shape[1])
                new_embedding.normal_(mean=0.0, std=0.02)
                
                # Copy over common tokens (assuming same special tokens)
                min_size = min(old_vocab_size, new_vocab_size)
                new_embedding[:min_size] = old_embedding[:min_size]
                
                checkpoint['text_embedding.weight'] = new_embedding
                
                # Resize text head as well
                if 'text_head.weight' in checkpoint:
                    old_head = checkpoint['text_head.weight']
                    new_head = torch.randn(new_vocab_size, old_head.shape[1])
                    new_head.normal_(mean=0.0, std=0.02)
                    new_head[:min_size] = old_head[:min_size]
                    checkpoint['text_head.weight'] = new_head
        
        # Load state dict
        model.load_state_dict(checkpoint, strict=False)
        
        # Move to device
        model = model.to(self.device)
        
        print("Model loaded successfully")
        return model
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_data_loader(
        self,
        manifest_file: str,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader"""
        dataset = GermanTTSDataset(
            manifest_file=manifest_file,
            tokenizer=self.tokenizer,
            mel_extractor=self.mel_extractor,
            max_text_length=self.config['gpt']['max_text_tokens'],
            max_mel_length=self.config['gpt']['max_mel_tokens']
        )
        
        def collate_fn(batch):
            # Pad sequences
            text_tokens = [item['text_tokens'] for item in batch]
            mel_spectrograms = [item['mel_spectrogram'] for item in batch]
            
            # Pad text tokens
            max_text_len = max(len(tokens) for tokens in text_tokens)
            padded_text_tokens = []
            text_attention_masks = []
            
            for tokens in text_tokens:
                pad_len = max_text_len - len(tokens)
                if pad_len > 0:
                    padded_tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
                    attention_mask = torch.cat([torch.ones(len(tokens)), torch.zeros(pad_len)])
                else:
                    padded_tokens = tokens
                    attention_mask = torch.ones(len(tokens))
                
                padded_text_tokens.append(padded_tokens)
                text_attention_masks.append(attention_mask)
            
            # Pad mel spectrograms
            max_mel_len = max(mel.shape[-1] for mel in mel_spectrograms)
            padded_mel_spectrograms = []
            mel_attention_masks = []
            
            for mel in mel_spectrograms:
                pad_len = max_mel_len - mel.shape[-1]
                if pad_len > 0:
                    padded_mel = torch.cat([mel, torch.zeros(mel.shape[0], pad_len)], dim=-1)
                    attention_mask = torch.cat([torch.ones(mel.shape[-1]), torch.zeros(pad_len)])
                else:
                    padded_mel = mel
                    attention_mask = torch.ones(mel.shape[-1])
                
                padded_mel_spectrograms.append(padded_mel)
                mel_attention_masks.append(attention_mask)
            
            return {
                'text_tokens': torch.stack(padded_text_tokens),
                'text_attention_masks': torch.stack(text_attention_masks),
                'mel_spectrograms': torch.stack(padded_mel_spectrograms),
                'mel_attention_masks': torch.stack(mel_attention_masks),
                'texts': [item['text'] for item in batch],
                'audio_ids': [item['audio_id'] for item in batch]
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def train(
        self,
        train_manifest: str,
        val_manifest: str,
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        gradient_clip_val: float = 1.0,
        save_every: int = 1000,
        log_every: int = 100,
        use_wandb: bool = False
    ):
        """Train the model"""
        
        if use_wandb:
            wandb.init(
                project="indextts2-german-finetuning",
                config={
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "use_lora": self.use_lora,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha
                }
            )
        
        # Create data loaders
        train_loader = self.create_data_loader(train_manifest, batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_manifest, batch_size, shuffle=False)
        
        # Setup optimizer
        if self.use_lora:
            # Only optimize LoRA parameters
            trainable_params = self.lora_manager.get_lora_parameters()
            # Also include text embedding and speed embedding
            trainable_params.extend(list(self.model.text_embedding.parameters()))
            trainable_params.extend(list(self.model.speed_emb.parameters()))
        else:
            # Fine-tune all parameters
            trainable_params = list(self.model.parameters())
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                text_tokens = batch['text_tokens'].to(self.device)
                text_attention_masks = batch['text_attention_masks'].to(self.device)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Create input sequence: [mel_tokens, text_tokens]
                # For now, we'll use a simplified approach
                # In practice, you'd need to implement the full training logic
                
                # Compute loss (simplified - you'd need to implement proper loss computation)
                loss = self._compute_loss(
                    text_tokens, text_attention_masks,
                    mel_spectrograms, mel_attention_masks
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_val)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % log_every == 0:
                    self.logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                    if use_wandb:
                        wandb.log({"train_loss": loss.item(), "step": global_step})
                
                # Save checkpoint
                if global_step % save_every == 0:
                    self._save_checkpoint(global_step, epoch, loss.item())
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Validation
            val_loss = self._validate(val_loader)
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss/len(train_loader),
                    "val_loss": val_loss
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(global_step, epoch, val_loss, is_best=True)
        
        self.logger.info("Training completed!")
        if use_wandb:
            wandb.finish()
    
    def _compute_loss(self, text_tokens, text_attention_masks, mel_spectrograms, mel_attention_masks):
        """Compute training loss (simplified implementation)"""
        # This is a simplified loss computation
        # In practice, you'd need to implement the full IndexTTS2 training logic
        # including proper text-to-mel generation and loss computation
        
        # For now, return a dummy loss
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_tokens = batch['text_tokens'].to(self.device)
                text_attention_masks = batch['text_attention_masks'].to(self.device)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                mel_attention_masks = batch['mel_attention_masks'].to(self.device)
                
                loss = self._compute_loss(
                    text_tokens, text_attention_masks,
                    mel_spectrograms, mel_attention_masks
                )
                
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, step, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if self.use_lora:
            checkpoint['lora_state_dict'] = {
                name: adapter.state_dict() 
                for name, adapter in self.lora_manager.lora_adapters.items()
            }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved at step {step}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndexTTS2 for German")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--german_vocab", type=str, required=True, help="Path to German vocab file")
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to training manifest")
    parser.add_argument("--val_manifest", type=str, required=True, help="Path to validation manifest")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = GermanTTSFineTuner(
        config_path=args.config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        german_vocab_path=args.german_vocab,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Start training
    fine_tuner.train(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )


if __name__ == "__main__":
    main()
