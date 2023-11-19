from pathlib import Path

def get_config():
    """
    Returns a dictionary containing configuration parameters for the transformer model.

    Configuration Parameters:
        - batch_size (int): Batch size for training.
        - num_epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for optimization.
        - max_seq_len (int): Maximum sequence length for input and output.
        - embedding_dim (int): Dimensionality of embedding vectors.
        - num_heads (int): Number of attention heads in multi-head attention layers.
        - src_lang (str): Source language code.
        - tgt_lang (str): Target language code.
        - model_folder (str): Folder to save model weights.
        - model_basename (str): Basename for model weight files.
        - preload (str): Path to pre-trained model weights, if any.
        - tokenizer_file (str): File pattern for saving/loading tokenizers.
        - experiment_name (str): Directory for storing experiment-related data.

    Returns:
        - config (dict): Dictionary containing configuration parameters.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "learning_rate": 10**-4,
        "max_seq_len": 785,
        "embedding_dim": 512,
        "num_heads": 8,
        "src_lang": "en",
        "tgt_lang": "es",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Generates the file path for saving or loading model weights.

    Parameters:
        - config (dict): Configuration parameters.
        - epoch (str): Epoch number to include in the filename.

    Returns:
        - file_path (str): Complete file path for model weights.
    """
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)