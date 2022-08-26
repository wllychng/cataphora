# code to test how word by word perplexity works 

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from copy import deepcopy

# load models and etc. using from_pretrained() method of various classes
# other model names for different gpt2 sizes: 'gpt2-medium' 'gpt2-large' 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# set up the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# test tokens for "he" , "she"
# "he" token id: 258
he_token_id = tokenizer("he", return_tensors='pt').input_ids.squeeze().item()
print(f"he token id: {he_token_id}")

# "she" token id: 7091
she_token_id = tokenizer("she", return_tensors='pt').input_ids
print(f"she token id: {she_token_id}")

sentence_str = "Before John goes to the store, he likes to take a shower."
tokenized_input = tokenizer(sentence_str, return_tensors='pt')
print(f"tokenized_input_ids, from dictionary key access: {tokenized_input['input_ids']}")
print(f"tokenized_input_ids, from class field access: {tokenized_input.input_ids}")

context_str = "Before John goes to the store,"
context = tokenizer(context_str, return_tensors = 'pt')
# from Davis and van Schijndel 2020, surprisal is negative log prob
# prob is softmax of "logits" 
# compare softmax prob of ID for "he" vs ID for "she"

# order of sentences: principleC_match, principleC_mismatch, noPrincipleC_match, noPrincipleC_mismatch
sentences = ["It seemed worrisome to him that John", "It seemed worrisome to her that John", "It seemed worrisome to his family that John", "It seemed worrisome to her family that John"]

tokenized_sents = tokenizer(sentences[0], return_tensors = 'pt')
print(f"size of tokenized sents: {tokenized_sents.input_ids.size()}")

with torch.no_grad():
	output = model(**tokenized_sents)
	
surps = torch.log2(torch.exp(-1*torch.nn.functional.log_softmax(output.logits, -1)))
# surps is a tensor of size batch x length x vocab

# get the token ID of "John" as the index, pull out surp
john_idx = 
print(f"surprisal for John: {surps[0][-1][john_idx]}")

print(f"tokenized input: {tokenized_input}")
tokenized_input_ids = tokenized_input
#print(f"size of tokenized input pytorch tensor: {tokenized_input.size()}")

# test on tensor input_ids types
#Y = tokenized_input["input_ids"]
#print(f"type of outermost tensor: {type(Y)}")
#print(f"type of tensor[0]: {type(Y[0])}")
#print(f"type of tensor[0][0]: {type(Y[0][0])}")

# test deep copy of input
X = deepcopy(tokenized_input)
print(f"size of initial input_ids: {X['input_ids'].size()}")

# indexing tensors is the same as python lists
# returned pt tensor has extra list layer e.g. [[1, 2, 3]]

# modify the tensor values in the innermost part
# X["input_ids"] = torch.reshape(torch.tensor(X["input_ids"][0][0:2]))

new_input_ids = X["input_ids"][0][0:2]
new_input_ids = torch.unsqueeze(new_input_ids, 0)
print(f"new input ids: {new_input_ids}")
print(f"size of new input ids: {new_input_ids.size()}")

print(f"old input: {tokenized_input})")
print(f"new input: {X})")

# to decode pt tensors, have to unwrap one layer of list, hence the [0] indexing
# note that the decoded text is a string, not a list of strings (tokens)
token_text = tokenizer.decode(tokenized_input["input_ids"][0])
print(f"token text: {token_text}")

# make labels
# target_ids = 

# use torch.no_grad() when doing predictions
with torch.no_grad():
	output = model(**tokenized_input)

# GPT2 output is a pair, has two output components: loss and logits
print(f"output.logits: {output.logits}")
print(f"type of output.logits: {type(output.logits)}")
print(f"size of output.logits: {output.logits.size()}")

# surprisals, from Davis and van Schjindel 2020 code
# size/shape of logits/surprisals: batch size x sequence length x vocab
surps = torch.log2(torch.exp(-1*torch.nn.functional.log_softmax(output.logits, -1)))
print(f"size of surps: {surps.size()}")

# perplexity: assumes nlls contains a list of the negative log likelihoods of each word
# torch.exp(), returns a new tensor with exponentials of input tensor
 
