import os
import utils
import shutil

#################### PARAMETRI  #########################

ORIGINAL_DIR = "dataset/train/casco_NO"
AUGMENTED_DIR = "dataset_augm/train/casco_NO"

#######################################################


# Copia immagini originali nella cartella
for img_name in os.listdir(ORIGINAL_DIR) :
    ### Controlli validità
    if (img_name == ".DS_Store") : continue # (...se è un file di macOS)  # noqa: E701
    if (os.path.isdir(os.path.join(ORIGINAL_DIR, img_name))) : continue # (...se è una sotto cartella)  # noqa: E701
    
    shutil.copyfile(os.path.join(ORIGINAL_DIR, img_name),
                    os.path.join(AUGMENTED_DIR, img_name))
        
        
# Crea immagini aumentate (luminosità) e copia nella cartella
utils.brightness_jitter(ORIGINAL_DIR, AUGMENTED_DIR)

# ... dalla stessa cartella ri-raddoppia le immagini (crop-flip-ecc) 
utils.resize_flip_rota_crop__directory(AUGMENTED_DIR, AUGMENTED_DIR)

        
