import torch
import torch.nn.functional as F

# Character encoding utilities
def create_mappings(names):
    chs = {ch for name in names for ch in name}
    s_to_i = {ch: i for i, ch in enumerate(sorted(chs), 1)}
    s_to_i['.'] = 0
    i_to_s = {v: k for k, v in s_to_i.items()}
    return s_to_i, i_to_s, chs

# Load and preprocess data
file_path = input("Enter the file path containing names: ")
with open(file_path, 'r') as f:
    names = f.read().splitlines()

s_to_i, i_to_s, chs = create_mappings(names)

xs_ch1, xs_ch2, y = [], [], []
for name in names:
    name = f".{name}."
    for ch1, ch2, ch3 in zip(name, name[1:], name[2:]):
        xs_ch1.append(s_to_i[ch1])
        xs_ch2.append(s_to_i[ch2])
        y.append(s_to_i[ch3])

xs_ch1 = torch.tensor(xs_ch1)
xs_ch2 = torch.tensor(xs_ch2)
y = torch.tensor(y)
elems = xs_ch1.nelement()

# Initialize weights
W = torch.randn((54, 27), requires_grad=True)

# Training settings
epochs = int(input("Enter number of epochs to train: "))

# Training loop
for epoch in range(1, epochs + 1):
    xs_ch1_enc = F.one_hot(xs_ch1, num_classes=27).float()
    xs_ch2_enc = F.one_hot(xs_ch2, num_classes=27).float()
    xs_enc = torch.cat([xs_ch1_enc, xs_ch2_enc], dim=1)

    logits = xs_enc @ W
    probs = F.softmax(logits, dim=1)

    loss = -probs[torch.arange(elems), y].log().mean() + 0.015 * (W**2).mean()

    W.grad = None
    loss.backward()
    W.data += -10 * W.grad

    if epoch % (epochs // 10) == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete!\n")

# Interactive prediction
print("Enter a letter to generate names (type '.' or other invalid input to exit):")

while True:
    usr_in = input("\nEnter first character of the name: ").lower()
    if usr_in not in chs:
        print("--------\nGoodbye!")
        break

    word = f'.{usr_in[-1]}'

    while True:
        xs_ch1_enc = F.one_hot(torch.tensor(s_to_i[word[-2]]), num_classes=27).float()
        xs_ch2_enc = F.one_hot(torch.tensor(s_to_i[word[-1]]), num_classes=27).float()
        xs_enc = torch.cat([xs_ch1_enc, xs_ch2_enc])

        logits = xs_enc @ W
        probs = F.softmax(logits, dim=0)

        sample_val = torch.multinomial(probs, 1).item()
        next_char = i_to_s[sample_val]

        if next_char == '.':
            break
        word += next_char

    print(f"Generated name: {word[1:]}")
