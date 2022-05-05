import argparse
import os
import logging
from src.utils.common import read_yaml
import tensorflow as tf


STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a")


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    params = config['params']
    PARENT_DIR = os.path.join(config['data']['local_dir'], config['data']['parent_data_dir']) 

    #get data ready
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        validation_split=params['validation_split'],
        subset="training",
        seed=params['seed'],
        image_size=params['image_shape'][:2],
        batch_size=params['batch_size'])

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        validation_split=params['validation_split'],
        subset="validation",
        seed=params['seed'],
        image_size=params['image_shape'][:2],
        batch_size=params['batch_size'])

    train_ds = train_ds.prefetch(buffer_size=params['buffer_size'])
    val_ds = val_ds.prefetch(buffer_size=params['buffer_size'])

    #load base model
    path_to_model_dir = config['model']['model_dir']
    path_to_model = os.path.join(path_to_model_dir, 'init_model.h5')
    classifier = tf.keras.models.load_model(path_to_model)
    logging.info(f"initial model loded which was saved at {path_to_model}")

    #start training
    EPOCHS = params['EPOCHS']
    logging.info("Training Started!!")
    classifier.fit(train_ds, epochs=EPOCHS, validation_data = val_ds)
    
    #save model
    path_to_model_dir = config['model']['model_dir']
    path_to_trained_model = os.path.join(path_to_model_dir, config['model']['trained_model'])
    classifier.save(path_to_trained_model)
    logging.info(f"Training completed, model saved at {path_to_trained_model}!!")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    # args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e