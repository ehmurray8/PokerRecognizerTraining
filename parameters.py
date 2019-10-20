# Name of the CoreML model to export the data as
COREML_MODEL_NAME = "Cards.mlmodel"

# Set this to the location where you would like to use the CoreML model in an ios project
OUTPUT_LOCATION = f"/Users/emurray/source_code/PokerRecognizerApp/PokerRecognizerApp/{COREML_MODEL_NAME}"

# The amount of the data to use as the training data when the training data is split
TRAIN_TEST_SPLIT = 0.8

# Number of GPUS to use while training the model [0=None, -1=All]
NUMBER_OF_GPUS = -1 

CACHE_DIRECTORY = "/home/emmet/source_code/PokerRecognizerTraining"
