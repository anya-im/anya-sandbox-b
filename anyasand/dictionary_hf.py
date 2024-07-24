import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer
from bitnet import replace_linears_in_hf
from .dictionary import Dictionary


class DictionaryHf(Dictionary):
    def __init__(self, db_path, pre_trained_model="line-corporation/line-distilbert-base-japanese"):
        super().__init__(db_path)
        self._tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, trust_remote_code=True)
        self._hf_model = AutoModel.from_pretrained(pre_trained_model)
        replace_linears_in_hf(self._hf_model)

        self._vec_size = 768

    def get_sword(self, wid):
        return np.concatenate([self._vector(str(wid)), self.vec_eye(wid)])

    def _vector(self, idx):
        with torch.no_grad():
            token = self._tokenizer(self._words[idx]["name"], return_tensors="pt")
            token_len = len(token["input_ids"].squeeze()) - 1
            outputs = self._hf_model(**token)
            vec = torch.zeros_like(outputs["last_hidden_state"].squeeze()[0]).detach()
            for i in range(token_len):
                if i == 0:
                    continue
                vec += outputs["last_hidden_state"].squeeze()[i]
            vec = torch.where(vec > 0., 1., 0.)
        return vec.numpy()
