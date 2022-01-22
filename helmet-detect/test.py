import json
import cv2
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_generators import SSD_VOC_DATA_GENERATOR
from networks import SSD300_VGG16
from losses import SSD_LOSS
from tensorflow.keras.callbacks import ModelCheckpoint

with open("configs/ssd300_vgg16.json") as config_file:
    config = json.load(config_file)

test_samples = []
with open(config["test"]["data"]["test_split_file"], "r") as split_file:
    lines = split_file.readlines()
    for line in lines:
        filename = line.split(" ")[0]
        image_file = os.path.join(config["test"]["data"]["images_dir"], f"{filename}.jpg")
        label_file = os.path.join(config["test"]["data"]["labels_dir"], f"{filename}.xml")
        sample = f"{image_file} {label_file}"
        test_samples.append(sample)

# data_generator = SSD_VOC_DATA_GENERATOR(
#     samples=["data/test.jpg data/test.xml"],
#     config=config
# )
model = SSD300_VGG16(config)
model.load_weights('cp_02_12.5455.h5')

model.compile(
        optimizer=Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam"
        ),
        loss=SSD_LOSS(
            alpha=config["training"]["alpha"],
            min_negative_boxes=config["training"]["min_negative_boxes"],
            negative_boxes_ratio=config["training"]["negative_boxes_ratio"]
        ).compute
    )


results = model.evaluate(
        x=SSD_VOC_DATA_GENERATOR(samples=test_samples, config=config),
        batch_size=2,
        # epochs=config["training"]["epochs"],
        # steps_per_epoch=len(training_samples)//config["training"]["batch_size"],
        # validation_data=SSD_VOC_DATA_GENERATOR(samples=validation_samples, config=config),
        # validation_steps=len(validation_samples)//config["training"]["batch_size"],
        callbacks=[
            ModelCheckpoint(
                'cp_{epoch:02d}_{loss:.4f}.h5',
                mode='min',
                monitor='loss',
                # save_weights_only=True,
                verbose=1,
            ),
        ]
    )
#tf.print('Accuracy: ', results[1]*100)
print(results)

limit = 1

for i, (batch_x, batch_y) in enumerate(data_generator):
    print(f"batch {i+1}")
    for j in range(len(batch_x)):
        print(f"-- item {j}")
    if i >= limit:
        break
