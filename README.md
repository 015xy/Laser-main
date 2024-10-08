# Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Mixture of Collaborative Experts

### Installation

The `./requirements.txt` list all Python libraries that Laser depends on, and you can install using:

```
conda create -n laser python=3.9
conda activate laser
pip install -r requirements.txt
```

### Datasets
In this work, we use 3 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) to train and evaluate our Laser:

- `Industrial and Scientific`
- `Arts, Crafts and Sewing`
- `Pet Supplies`

You can process these data using the script `finetune_data/process.py` and running the following commands:
```bash
cd data
python process.py --meta_file_path META_PATH --file_path SEQ_PATH --output_path OUTPUT_FOLDER
```



### Model
In this work, we use the `ChatGLM2-6B` model, which can be downloaded from [link](https://huggingface.co/THUDM/chatglm2-6b) and inserted to `./Translator/models` 

```
cd ./Translator/models
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```


### Run
training stage 1

```
cd Translator/train
python train.py --cfg-path ./pretrain_stage1.yaml
```

training stage 2

```
python train.py --cfg-path ./pretrain_stage2.yaml
```
