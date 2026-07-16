from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.set_printoptions(sci_mode=False)

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
outputdir = "./Model"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True)


#weights
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data.contiguous().numpy().tofile(f'{outputdir}/{name}')
        print(f"Saved {name} to {outputdir}/{name}")

#tokenizer
tokenizer.save_pretrained("./Model/tokenizer")