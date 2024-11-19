import asyncio
# import logging
from typing import Text

import rasa.utils.common
from rasa.model_testing import test_nlu


# logger = logging.getLogger(__name__)

def train(
        domain_file_path,
        config_file_path,
        training_file_path: Text,
        output_dir,
) -> None:
    """
    Trains a model.
    :param domain_file_path: Domain file path
    :param config_file_path: Configuration file path
    :param training_file_path: Training data file path
    :param output_dir: Output directory for the model
    :return:
    """
    rasa.api.train(domain=domain_file_path, config=config_file_path, training_files=training_file_path,
                   output=output_dir)

    return


def test(
        model_path,
        test_file_path,
        output_dir,
) -> None:
    """
    Test a trained model.

    :param model_path: The model path
    :param test_file_path: The test file path
    :param output_dir: The output directory
    """
    asyncio.run(
        test_nlu(
            model_path,
            test_file_path,
            output_dir,
            {'successes': True, 'errors': True, 'quiet': True})
    )

    return
