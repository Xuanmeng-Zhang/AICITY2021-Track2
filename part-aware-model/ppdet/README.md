# Train model:
source train_aicity.sh

# Export inference model:
source export_aicity_model.sh

# Inference model:
source inference.sh 
Inference results will be saved in folder 'aicity_crop_data', these results will be used in the following training steps of the part feature extract models.