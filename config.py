# File config
FILE_PATH = "eng_to_spa.txt"

# Model Config
TinyGPT_Config = {
    "VOCAB_SIZE": 8000,
    "EMBEDDING_SIZE": 8003,
    "D_MODEL": 512,
    "D_FF": 2048,
    "DROPOUT": 0.2,
    "N_HEADS": 8,
    "N_LAYERS": 6,
    "EPS": 10**-6,
    "QKV_BIAS": False,
}

# Training Config
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
VOCAB_SIZE = 8000
