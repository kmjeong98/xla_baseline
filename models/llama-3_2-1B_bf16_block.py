from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "meta-llama/Llama-3.1-8B", 32

def get_model(): return HFWrapper(transformers.AutoModel.from_pretrained(ID, torch_dtype=torch.bfloat16).eval())
def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(ID)
    vocab = tok.vocab_size
    ids = torch.randint(0, vocab, (1, _SEQ), dtype=torch.long)
    return ids, torch.ones_like(ids)

