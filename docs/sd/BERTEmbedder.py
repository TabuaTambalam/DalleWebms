import torch
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
max_length = 77
dedup=dict()
transformer = torch.jit.load('transformer_pnnx.pt')
embedding = torch.nn.Embedding.from_pretrained(transformer.text_model_embeddings_token_embedding.weight)

def tok(text, pad=False):
  padstr='do_not_pad'
  if pad:
    padstr='max_length'
  batch_encoding = tokenizer(text, truncation=True, max_length=max_length, return_length=True,
                        return_overflowing_tokens=False, padding=padstr, return_tensors='pt')

  return batch_encoding['input_ids'][0]

emp=tok('')
tok_bos, tok_eos = int(emp[0]), int(emp[1])
emp=embedding(emp)
emb_bos, emb_eos = emp[0] ,emp[1]

def encode(text):
  tokenz=tok(text)
  l_tok=tokenz.size(0)
  poz_dict=[None]*l_tok
  for i in range(l_tok):
    toki=int(tokenz[i])
    poz_dict[i]=(tokenizer.decode(toki),toki )
  emp=torch.cat(( embedding(tokenz),emb_eos.expand(max_length-l_tok,-1) )).unsqueeze(0)

  return transformer(emp)[:,:l_tok,:],poz_dict
