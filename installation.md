```
conda create -p /scratch/bbsg/shengcao/conda_envs/sd_llava python=3.10
conda activate /scratch/bbsg/shengcao/conda_envs/sd_llava
module load cuda/11.8.0
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

cd LLaVA
pip install --upgrade pip
pip install -e ".[train]" # removed pytorch dependencies in pyproject.toml
pip install flash-attn --no-build-isolation

pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers[torch]==0.15.0
pip install jupyterlab  # maybe not needed
pip install ipympl tensorboardX

cd segment-anything
pip install -e .
pip install opencv-python pycocotools pycocoevalcap matplotlib spacy
python -m spacy download en_core_web_trf en_core_web_lg

# for evaluation
pip install openpyxl openai==0.28
```

Consider adding this patch? https://github.com/huggingface/diffusers/pull/3076