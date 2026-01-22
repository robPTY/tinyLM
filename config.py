# File config
FILE_PATH = "data/eng_to_spa.txt"

# Model Config
TinyLM_Config = {
    "VOCAB_SIZE": 8000,
    "EMBEDDING_SIZE": 8003,
    "D_MODEL": 512,
    "D_FF": 2048,
    "DROPOUT": 0.1,
    "N_HEADS": 8,
    "N_LAYERS": 6,
    "EPS": 10**-6,
    "QKV_BIAS": False,
}

# Training Config
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
TOTAL_VOCAB_SIZE = 8000
BETA1 = 0.9
BETA2 = 0.98
LEARNING_RATE = 10**-4
EPS = 10**-9
E_LS = 0.1 # For label smoothing
EPOCHS = 10
MODEL_FOLDER = "weights"
MODEL_BASENAME = "tinyLM"
