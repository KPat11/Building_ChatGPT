import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Finding unique characters for encoding

with open('homer.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset: ", len(text))

#Looking at first 1000 characters
print(text[:1001])

# unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


# 2. Tokenization: Encoding and Decoding Strategy
'''I will be mapping characters to numbers and create functions to encode and decode. 
I know their are other methods like Sentencepiece or a byte-pair tokenizer like tiktoken which openai uses 
but I elected to code out instead of using libraries for learning/practice purposes.'''

# Mapping
encode_map = { ch:i for i,ch in enumerate(chars) }
decode_map = { i:ch for i, ch in enumerate(chars) }

#encoder takes string and maps to list of integers
encode = lambda e: [encode_map[c] for c in e]
#decode takes list of integers and outputs a string
decode = lambda d: ''.join([decode_map[u] for u in d])

print(encode("hellooo friend"))
print(decode(encode("hellooo friend")))

# encoding entire dataset using pytorch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1001]) #peek at first 1000 characters

# # 3. Split into Train/Test
# Splitting train/test, chunk definitions, and batching for multiple chunks at same time.

#taking 90% of data for train, and rest for validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(train_data)
print(val_data)

# will be training random chunks rather than every line for computation reasons
chunk_size = 8
train_data[:chunk_size+1]


# Setting up next likely value logic and sanity checking

x = train_data[:chunk_size]
y = train_data[1:chunk_size+1]
for i in range(chunk_size):
    context = x[:i+1]
    target = y[i]
    print(f"When input is {context} the target is: {target}")


# manual seed for random generator for this code if you would like to reproduce results
#torch.manual_seed(1337)
batch_size = 4 # how many chunks we will process at once
chunk_size = 8 # max context length for predictions

def get_batch(split):
    #generating a small batch of data of inputs x and targets y
    data = train_data if split== 'train' else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,)) # x position for random batch
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    return x,y

x_batch, y_batch = get_batch('train')
print('inputs:')
print(x_batch.shape)
print(x_batch)
print('targets:')
print(y_batch.shape)
print(y_batch)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(chunk_size): # time dimension
        context = x_batch[b, :t+1]
        target = y_batch[b,t]
        print(f"When input is {context.tolist()} the target is: {target}")


# 4. Neural Network
'''Now that the data is prepared into train/validatin sets and batching, randomized positioning has been defined, 
 and we have encoded those batches.I will now implement a neural network with the data.'''

# Defining module for a simple Bigram Language Model

'''
 1. Creating token embedding tables for positional reference
 2. Creating a embedding table for multidimensional tensor for token pairs (B,T,C)
 3. Defining loss function (cross_entorpy) and making sure it aligns with expected loss (-ln(1/88)) -- 88 being the number of unique characters (vocab_size variable)
 4. Generate function will get predictions, apply the softmax activation function to find probability most likely next character, 
 and append the character with the highest probability to the end of the running sequence.
 
 Simple model and progress is made but will need to implement context and transformsers.'''
#torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # creating an embedding table for position reference to block out current positions (similar to a visited table)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) token pairing for tensor
        # due to pytorch cross entropy wanting the tensor to be (B,C,T) we will need to reshape
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # loss definition will be cross entropy
            loss = F.cross_entropy(logits, targets) # expecting loss of 4.477368 (-ln(1/88))

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            #focusing on last step 'time' step
            logits = logits[:, -1, :] # becomes (B, C)
            # using softmax as activation function
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B, 1)
            # appending sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
logits, loss = m(x_batch, y_batch)
print(logits.shape)
print(loss)

print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long),max_new_tokens=100)[0].tolist()))

''' First run of model: 
 torch.Size([32, 88])
 tensor(4.9037, grad_fn=<NllLossBackward0>)
 
 7-OW rt—60uL(EQzCcrraUtm(hTpb#GQ%Kgbq#﻿3BA9•AkGgcBP-RdK
 r“5Oyf(W?C;K/keSI
 0yPtYbJ’”8zCz#UJ(:,J5••AZR '''

# Which is expected as it is random -- training is yet to be done

# Optimizer - using Adam and using higher learning rate because of smaller sample data
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

for steps in range(10000):
    #sample batch
    x_batch,y_batch = get_batch('train')

    #eval loss
    logits, loss = model(x_batch,y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# Model results after optimizing -- progress but still needs work.
print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long),max_new_tokens=100)[0].tolist()))



