# TSAM

The code for paper: 《TSAM: A Two-Stream Attention Model for Causal Emotion Entailment》


## Requirements

- Python 3.6
- PyTorch 1.6
- [Transformers from Hugging Face](https://github.com/huggingface/transformers)

With Anaconda, we can create the environment with the provided `environment.yml`:

```bash
conda env create --file environment.yml 
conda activate CEE
```

The code has been tested on Ubuntu 16.04 using a single GPU. For multi-GPU training, a little extra work may be needed, and please refer to these examples: [Hugging Face](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) and [CLUEbenchmark](https://github.com/CLUEbenchmark/CLUE/tree/master/baselines/models_pytorch). 

<br>

## Run Steps

1. `data/` including [original annotation dataset](https://github.com/soujanyaporia/RECCON) and our preprocessed dataset.
2. `code/` including our codes. 
3. Download necessary pre-trained model files (RoBERTa base and large).
4. Configure hyper-parameters in `code/config.py`
5. Run our model:

```bash
python code/main.py
```
