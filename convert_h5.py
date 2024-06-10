import tensorflow as tf

# Charger le modèle à partir du fichier fourni
model_path = "mask_detector.model"
model = tf.keras.models.load_model(model_path)

# Sauvegarder le modèle au format H5
h5_model_path = "mask_detector.h5"
model.save(h5_model_path)

h5_model_path
