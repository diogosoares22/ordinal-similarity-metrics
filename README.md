# ordinal-similarity-metrics
Repository contains the code for a ordinal similarity metrics called Triplet Similarity Index (TSI) and Quadruplet Similarity Index (QSI)

conda create -n ordinal-similarity-metrics python=3.11

conda activate ordinal-similarity-metrics

pip install -e .

gdown "https://drive.google.com/uc?id=1pksaioLn57UmbeKm7gMIkfSdqNdA7a4m" -O data/cifar-10-final-epoch-val-representations.npz

cd data

wget http://images.cocodataset.org/zips/train2017.zip -O data/coco/train2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations/trainval2017.zip

