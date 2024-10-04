import os
from PIL import Image
import torchvision.transforms as transforms
import numpy
import cv2



def brightness_jitter(source_image_dir, result_dir, BRIGHT_JITTER=0.5) :
    
      for image_name in os.listdir(source_image_dir) :
        
        ### Controlli validità
        if (image_name == ".DS_Store") : continue # (...se è un file di macOS)  # noqa: E701
        if (os.path.isdir(os.path.join(source_image_dir, image_name))) : continue # (...se è una sotto cartella)  # noqa: E701
        
        # Read image.
        img_PIL = Image.open(os.path.join(source_image_dir, image_name))
        
        # Trasform 
        transform = transforms.ColorJitter(brightness=BRIGHT_JITTER)
        img_trans_PIL = transform(img_PIL)
        
        # Convert in cv2 to save
        img_trasf_cv2 = numpy.array(img_trans_PIL)[:, :, ::-1].copy()
        
        # Save
        cv2.imwrite(os.path.join(result_dir, image_name.split(".")[0]+"_bright.jpg"), img_trasf_cv2)


def resize_flip_rota_crop__directory(image_dir, result_dir, RESIZE_VALUE=224, TRESH_CROPPING=1.2) :
   
    for image_name in os.listdir(image_dir):
        
        ### Controlli validità
        if (image_name == ".DS_Store") : continue # (...se è un file di macOS)  # noqa: E701
        if (os.path.isdir(os.path.join(image_dir, image_name))) : continue # (...se è una sotto cartella)  # noqa: E701
        
        image_path = os.path.join(image_dir, image_name)
        
        aug_image = resize_flip_rot_crop_image(image_path, RESIZE_VALUE, TRESH_CROPPING)
        
        aug_image.save(f"{result_dir}/{image_name.split('.')[0]}_augm.{image_name.split('.')[1]}")


def resize_flip_rot_crop_image(image_path, RESIZE_VALUE=224, TRESH_CROPPING=1.2):
    image = Image.open(image_path)
    
    ### Per riempimento rotazione
    data = numpy.asarray(image)
    primo_pixel = data[0][0][:]

    ### Per cropping
    # w_resized = RESIZE_VALUE
    # h_resized = round(RESIZE_VALUE * image.size[1] / image.size[0])

    transform = transforms.Compose([
        # transforms.Resize(RESIZE_VALUE),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomRotation((-15,15), fill=(primo_pixel[0],primo_pixel[1], primo_pixel[2])),
        transforms.RandomCrop((round(image.size[1]/TRESH_CROPPING), round(image.size[0]/TRESH_CROPPING))),
    ])
    
    trasm_image = transform(image)
    return trasm_image