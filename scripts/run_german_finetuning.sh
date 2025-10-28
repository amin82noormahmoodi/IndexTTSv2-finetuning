#!/bin/bash

# German IndexTTS2 Fine-tuning Pipeline
# This script runs the complete fine-tuning workflow

set -e  # Exit on any error

# Configuration
DATA_DIR="german_sample_data"
METADATA_FILE="german_sample_data/metadata.csv"
PROCESSED_DIR="processed_german_data"
CHECKPOINT_DIR="checkpoints/german"
CONFIG_FILE="configs/german_config.yaml"
VOCAB_SIZE=16000
MAX_FILES=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if required files exist
check_prerequisites() {
    log "Checking prerequisites..."
    
    if [ ! -d "$DATA_DIR" ]; then
        error "Data directory $DATA_DIR not found"
        exit 1
    fi
    
    if [ ! -f "$METADATA_FILE" ]; then
        error "Metadata file $METADATA_FILE not found"
        exit 1
    fi
    
    if [ ! -f "checkpoints/gpt.pth" ]; then
        error "Pre-trained model checkpoints/gpt.pth not found"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Step 1: Data Preprocessing
preprocess_data() {
    log "Step 1: Preprocessing German data..."
    
    python scripts/prepare_german_data.py \
        --data_dir "$DATA_DIR" \
        --metadata_file "$METADATA_FILE" \
        --output_dir "$PROCESSED_DIR" \
        --max_files "$MAX_FILES" \
        --vocab_size "$VOCAB_SIZE"
    
    if [ $? -eq 0 ]; then
        success "Data preprocessing completed"
    else
        error "Data preprocessing failed"
        exit 1
    fi
}

# Step 2: Create train/val split
create_splits() {
    log "Step 2: Creating train/validation splits..."
    
    python -c "
import json
import random
from pathlib import Path

# Load manifest
manifest_file = Path('$PROCESSED_DIR/train_manifest.jsonl')
with open(manifest_file, 'r') as f:
    data = [json.loads(line) for line in f]

# Shuffle data
random.seed(42)
random.shuffle(data)

# Split data (80% train, 10% val, 10% test)
n = len(data)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Save splits
with open('$PROCESSED_DIR/train_manifest.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('$PROCESSED_DIR/val_manifest.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('$PROCESSED_DIR/test_manifest.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'Train: {len(train_data)} samples')
print(f'Val: {len(val_data)} samples')
print(f'Test: {len(test_data)} samples')
"
    
    success "Data splits created"
}

# Step 3: Fine-tuning
finetune_model() {
    log "Step 3: Fine-tuning model..."
    
    # Create checkpoint directory
    mkdir -p "$CHECKPOINT_DIR"
    
    python scripts/finetune_german.py \
        --config "$CONFIG_FILE" \
        --model_path "checkpoints/gpt.pth" \
        --output_dir "$CHECKPOINT_DIR" \
        --german_vocab "$PROCESSED_DIR/german_bpe.model" \
        --train_manifest "$PROCESSED_DIR/train_manifest.jsonl" \
        --val_manifest "$PROCESSED_DIR/val_manifest.jsonl" \
        --use_lora \
        --lora_rank 16 \
        --lora_alpha 16.0 \
        --num_epochs 10 \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --use_wandb
    
    if [ $? -eq 0 ]; then
        success "Model fine-tuning completed"
    else
        error "Model fine-tuning failed"
        exit 1
    fi
}

# Step 4: Evaluation
evaluate_model() {
    log "Step 4: Evaluating model..."
    
    # Create evaluation directory
    mkdir -p "evaluation_results"
    
    # Generate sample audio
    python scripts/evaluate_german.py \
        --config "$CONFIG_FILE" \
        --model_path "$CHECKPOINT_DIR/best_model.pt" \
        --german_vocab "$PROCESSED_DIR/german_bpe.model" \
        --output_dir "evaluation_results/samples" \
        --sample_texts \
            "Hallo, wie geht es Ihnen heute?" \
            "Das Wetter ist heute sehr schön." \
            "Ich freue mich, Sie kennenzulernen." \
            "Können Sie mir bitte helfen?" \
            "Vielen Dank für Ihre Hilfe."
    
    # Evaluate on test set
    python scripts/evaluate_german.py \
        --config "$CONFIG_FILE" \
        --model_path "$CHECKPOINT_DIR/best_model.pt" \
        --german_vocab "$PROCESSED_DIR/german_bpe.model" \
        --test_manifest "$PROCESSED_DIR/test_manifest.jsonl" \
        --output_dir "evaluation_results" \
        --reference_audio "examples/voice_01.wav"
    
    if [ $? -eq 0 ]; then
        success "Model evaluation completed"
    else
        warning "Model evaluation had issues, but continuing..."
    fi
}

# Step 5: Generate final report
generate_report() {
    log "Step 5: Generating final report..."
    
    cat > "evaluation_results/finetuning_report.md" << EOF
# German IndexTTS2 Fine-tuning Report

## Configuration
- **Dataset**: $DATA_DIR
- **Max Files**: $MAX_FILES
- **Vocabulary Size**: $VOCAB_SIZE
- **LoRA Rank**: 16
- **LoRA Alpha**: 16.0
- **Epochs**: 10
- **Batch Size**: 4
- **Learning Rate**: 1e-4

## Results
- **Model Checkpoint**: $CHECKPOINT_DIR/best_model.pt
- **German Vocab**: $PROCESSED_DIR/german_bpe.model
- **Sample Audio**: evaluation_results/samples/
- **Evaluation Results**: evaluation_results/evaluation_results.json

## Usage
\`\`\`python
from indextts.infer_v2 import IndexTTS2

# Load German model
tts = IndexTTS2(
    cfg_path="$CONFIG_FILE",
    model_dir="$CHECKPOINT_DIR"
)

# Generate German speech
audio = tts.infer_generator(
    spk_audio_prompt="reference_speaker.wav",
    text="Hallo, wie geht es Ihnen heute?",
    output_path="output.wav"
)
\`\`\`

## Files Generated
- \`$PROCESSED_DIR/\`: Preprocessed German data
- \`$CHECKPOINT_DIR/\`: Fine-tuned model checkpoints
- \`evaluation_results/\`: Evaluation results and samples
- \`$CONFIG_FILE\`: German-specific configuration

## Next Steps
1. Test the model with your own German text
2. Fine-tune further if needed
3. Deploy for production use
EOF
    
    success "Final report generated"
}

# Main execution
main() {
    log "Starting German IndexTTS2 fine-tuning pipeline..."
    
    # Check prerequisites
    check_prerequisites
    
    # Run pipeline steps
    preprocess_data
    create_splits
    finetune_model
    evaluate_model
    generate_report
    
    success "German fine-tuning pipeline completed successfully!"
    log "Check evaluation_results/ for results and samples"
    log "Model checkpoint: $CHECKPOINT_DIR/best_model.pt"
}

# Handle command line arguments
case "${1:-}" in
    "preprocess")
        check_prerequisites
        preprocess_data
        create_splits
        ;;
    "train")
        finetune_model
        ;;
    "evaluate")
        evaluate_model
        ;;
    "all"|"")
        main
        ;;
    *)
        echo "Usage: $0 [preprocess|train|evaluate|all]"
        echo "  preprocess: Only preprocess data"
        echo "  train: Only train the model"
        echo "  evaluate: Only evaluate the model"
        echo "  all: Run complete pipeline (default)"
        exit 1
        ;;
esac
