# satellite-imagery-POI

This project for TheWebConf2022 paper "Beyond the First Law of Geography: Learning Satellite Imagery Representations by Leveraging Point of Interests" and is mainly modified from the following github repository:
https://github.com/Spijkervet/SimCLR

The project consists of three parts: 
# 1. Constructing contrastive samples:

python constructing_contrastive_sample_POI.py

Input the satellite image name list and corresponding POI array.

# 2. Contrastive learning model training:

python main.py

Need to modify the hyperparameters in ./config/config.yaml and in dataloader.py.  
In our work, we use the Resnet-18 as the backbone.

# 3. Evaluation:

python feature_extract_simcrl.py   # Extract the image embeddings with the trained model.

python eval.py   # evaluation: attential fusion model.   Input the image embeddings and corresponding socioeconomic indicators
