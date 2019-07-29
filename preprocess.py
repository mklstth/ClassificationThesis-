import pydicom as dicom
import os
import cv2
import numpy as np
import PIL # optional

# For generation of separated validation data
# valid_generator = valid_datagen.flow_from_directory(
#     directory=r"./valid/",
#     target_size=(224, 224),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )

# switch it True for producing PNG format
PNG = False

# Specify the .dcm folder path
DICOM_path = "/media/mikes/Data/DATA/koponya/koponya_p20"

# Specify the output jpg/png folder path
OP_path = "/media/mikes/Data/DATA/valid/asd"

images_path = os.listdir(DICOM_path)
try:
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(DICOM_path, image))
        pixel_array_numpy = ds.pixel_array
        if PNG == False:
            image = image.replace('.dcm', '.jpg')
        else:
            image = image.replace('.dcm', '.png')
        cv2.imwrite(os.path.join(OP_path, image), pixel_array_numpy)

        # img = cv2.imread(os.path.join(OP_path, image), 0)
        # equ = cv2.equalizeHist(img)
        # res = np.hstack((img, equ))  # stacking images side-by-side
        # cv2.imwrite(os.path.join(OP_path, image), res)


        if n % 50 == 0:
            print('{} image converted'.format(n))
except:
    print("Could not read file:", image)

