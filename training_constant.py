# Set the config for training

BATCH_SIZE = 512
LIST_LABELS = ['normal', 'deep', 'strong']
N_CLASSES = len(LIST_LABELS)
N_EPOCHS = 1
# INPUT_SIZE = (40, 126, 1) # Input size for CNN training
INPUT_SIZE = (40, 126) # Input size for LSTM training
# TRAINING_SOURCE = 'D:/Do An/breath-deep/data/datawav_filter/train'
# VALID_SOURCE = 'D:/Do An/breath-deep/data/datawav_filter/test'

TRAINING_SOURCE = 'D:/Do An/Datasets/Breath_datasets_wav/Training/output/train'
VALID_SOURCE = 'D:/Do An/Datasets/Breath_datasets_wav/Training/output/test'

MODE = 'TRAINING'
MODEL_OUTPUT = './model_output'