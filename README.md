# ordinal-similarity-metrics
Repository contains the code for a ordinal similarity metrics called Triplet Similarity Index (TSI) and Quadruplet Similarity Index (QSI)

conda create -n ordinal-similarity-metrics python=3.11

conda activate ordinal-similarity-metrics

pip install -e .

gdown "https://drive.google.com/uc?id=1pksaioLn57UmbeKm7gMIkfSdqNdA7a4m" -O data/cifar-10-final-epoch-val-representations.npz

cd data

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar


