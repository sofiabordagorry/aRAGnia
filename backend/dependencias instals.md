(para que me ande a matias el cuda sino torch no detecta gpu) despoues de instalar las dependencias

python -m pip uninstall -y torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

(PARA MI PC NO FUNCIONA PORQUE NECESITO TORCH CU128 SI O SI POR MI GPU)

####################
awq:

pip install autoawq==0.2.6

pip uninstall -y transformers
pip install transformers==4.46.3
pip install numpy==1.26.4

####################
para # "google/gemma-4-E4B-it", # falla con transformers 4.x
python -m pip install -U "transformers>=5.5.0"

####################
gptq:
NO TODOS LOS MODELOS GPTQ ANDAN
pip uninstall -y peft
pip install peft==0.5.0
pip install auto-gptq==0.7.1
pip install optimum==1.21.4
