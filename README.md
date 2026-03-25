## Important! Read Before Using

In order to use this model on your own data, you will need to collect your data from Discord. You can request your data from Discord by following these steps (on desktop):
1. Go to your Discord settings
2. Click on "Data & Privacy"
3. Request your data by clicking the "Request Data" button. This will send a request to Discord, and they will prepare your data for download. It may take several days for them to process your request, depending on the amount of data you have.
4. When your data is ready, you will receive an email from Discord with a link to download your data. The data will be provided in a zip file. Download and extract the zip file to access your Discord messages.
5. Update the configuration in `config.yml` to point to the directory where you extracted your Discord data. The script will look for message files in this directory to collect and analyze.

## Installation

1. Clone this repository create a virtual environment (pip, uv, conda, etc.)

2. Install the required dependencies

```
pip install -r requirements.txt
```

3. (Optional) Install PyTorch with CUDA support for GPU training

```
pip install torch --index-url https://download.pytorch.org/whl/<cuda_version>
```

## Configuration

All settings are managed in `config.yml`. Key configuration sections:

- **model**: Base model, training parameters, and labels
- **dataset**: Dataset name and splits
- **paths**: Input/output file paths
- **processing**: Text cleaning patterns and thresholds
- **logging**: Logging configuration

## Usage

### Training

```
python train.py
```

This will train the model and save checkpoints to the `models/` directory.
~1 hour for 10 epochs on 8xA500 GPUs (200 TFLOPS) using the default configuration. Adjust `config.yml` for different settings or hardware.

### Basic Prediction

```
python predict.py
```

Enter text interactively to get toxicity predictions. Good for quick testing and debugging.

### Testing

```
python test.py
```

Runs evaluation on the test set and outputs average loss.

### Collect Discord Messages

```
python collect_msgs.py
```

This collects and cleans messages from the configured input directory.

### Analyze Messages

```
python analyze.py
```

This runs toxicity analysis on the collected messages and outputs statistics.
