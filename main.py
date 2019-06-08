import turicreate
from shutil import copy2
import os
import re

from constants import *
from parameters import COREML_MODEL_NAME, OUTPUT_LOCATION, TRAIN_TEST_SPLIT


# Defines the actions the program should take
SAVE_DATA = False
EXPORT_MODEL = True


def save_data():
    """
    Save the Training data to the disk in the SFrame format, in the current directory.
    """
    data = turicreate.image_analysis.load_images(TRAINING_DATA_NAME, with_path=True)

    data[CARD_NAME_LABEL] = data['path'].apply(lambda path: __path_to_label(path))
    data.save(SFRAME_NAME)
    print("Finished saving SFrame")


def __path_to_label(path: str) -> str:
    last_directory_name = os.path.split(os.path.dirname(path))[1]
    label_parts = re.sub('([a-z])([A-Z])', r'\1 \2', last_directory_name).split()
    label_parts[1] = label_parts[1].lower()
    return " ".join(label_parts)


def load_data() -> turicreate.SFrame:
    """
    Load the data from the disk, and return the SFrame object.

    :return: SFrame object loaded from the disk
    """
    return turicreate.SFrame(SFRAME_NAME)


def create_and_save_model(data: turicreate.SFrame):
    """
    Creates the CoreML model using the data SFrame, and saves it to the current directory.

    :param data: The SFrame that was created using the training data
    """
    train_data, test_data = data.random_split(TRAIN_TEST_SPLIT)
    model = turicreate.one_shot_object_detector.create(train_data, target=CARD_NAME_LABEL)
    _ = model.predict(test_data)
    metrics = model.evaluate(test_data)
    print(metrics[ACCURACY_LABEL])
    model.save(MODEL_NAME)
    model.export_coreml(COREML_MODEL_NAME)


if __name__ == "__main__":
    turicreate.config.set_num_gpus(0)

    # Uncomment when you need to create the SFrame
    if SAVE_DATA:
        save_data()

    DATA = load_data()
    create_and_save_model(DATA)

    # Copy the data model into the output location to be used in an app
    if EXPORT_MODEL:
        copy2(f'./{COREML_MODEL_NAME}', OUTPUT_LOCATION)
