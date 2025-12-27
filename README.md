# OuteTTS 1.0-1B on Replicate

Deploy [OuteAI/Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) to Replicate for cloud-based text-to-speech inference.

## Features

- **23 languages** including Persian/Farsi with native script support
- **~2GB model** (1B parameters, BF16)
- **Zero-shot voice cloning** capability (not yet exposed in API)

## Deployment

### 1. Create model on Replicate

Go to [replicate.com/create](https://replicate.com/create) and create a new model.

### 2. Add GitHub secret

Add `REPLICATE_API_TOKEN` to your repository secrets.

### 3. Trigger the workflow

Run the "Push to Replicate" workflow from the Actions tab, providing your model name (e.g., `yourusername/outetts-1b`).

## Usage

Once deployed, call via the Replicate API:

```python
import replicate

output = replicate.run(
    "yourusername/outetts-1b:latest",
    input={
        "text": "سلام، حال شما چطور است؟",
        "speaker": "EN-FEMALE-1-NEUTRAL",
        "temperature": 0.4
    }
)
print(output)  # URL to generated audio
```

## Local testing

```bash
# Install Cog
brew install cog  # or see https://cog.run

# Download weights (requires ~2GB)
cog run script/download_weights

# Test prediction
cog predict -i text="Hello, how are you?"
```
