# satellite-imagery-POI

This project for TheWebConf2022 paper "Beyond the First Law of Geography: Learning Satellite Imagery Representations by Leveraging Point of Interests".

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

python eval.py   # evaluation: attential fusion model.   Need the image embeddings and corresponding socioeconomic indicators.

# Descriptions for data:
1. "image_name.csv" and "POI_array_cate_1st_for_images.txt" are satellite image name list and corresponding POI array (For model training).

2. "dianping_aggre_Sat.csv", "pd_aggre_Sat.csv", and "population_aggre_Sat.csv" are respectively indicator data for number of comment, population density, and population count (Beijing), which are processed to map the corresponding satellite images. The reference of the original number of comment data is mentioned in the paper. And the original population data is from Worldpop project.
