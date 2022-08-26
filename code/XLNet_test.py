from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', output_hidden_states=True, return_dict=True, return_unused_kwargs=True)
model = XLNetModel.from_pretrained('xlnet-base-cased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

print(len(outputs))
print(type(outputs[0]))
print(type(outputs[1]))
