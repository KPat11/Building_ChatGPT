import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32 # how many chunks we will process at once
chunk_size = 8 # max context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------------------------------------------------

# if you would like to replicate, here is the random generator setting
torch.manual_seed(1337)

# Data used
with open('homer.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Mapping
encode_map = { ch:i for i,ch in enumerate(chars) }
decode_map = { i:ch for i, ch in enumerate(chars) }
#encoder takes string and maps to list of integers
encode = lambda e: [encode_map[c] for c in e]
#decode takes list of integers and outputs a string
decode = lambda d: ''.join([decode_map[u] for u in d])

# encoding entire dataset using pytorch
data = torch.tensor(encode(text), dtype=torch.long)

# Train/Test splits
n = int(0.9*len(data)) # taking 90% of data for train, and rest for validation
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    #generating a small batch of data of inputs x and targets y
    data = train_data if split== 'train' else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,)) # x position for random batch
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    return x,y

# telling pytorch to not do backpropagation for compute efficiency sake
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
m = model.to(device)

# First run of model was random and ununiformed, which is expected as it is random -- training is yet to be done

# Optimizer - using Adam and using higher learning rate because of smaller sample data
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # setting cadence of when to eval the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample batch of data
    x_batch, y_batch = get_batch('train')

    # eval loss
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long),max_new_tokens=100)[0].tolist()))



