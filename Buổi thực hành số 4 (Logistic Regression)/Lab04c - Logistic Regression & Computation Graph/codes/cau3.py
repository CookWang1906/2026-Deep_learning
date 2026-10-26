import torch

# Khai báo các biến với requires_grad=True
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(4.0, requires_grad=True)
d = torch.tensor(5.0, requires_grad=True)

# Định nghĩa hàm J
J = 3 * a + 2 * (4 * b + 5 * c * d)

# Lan truyền ngược
J.backward()

print(f"Giá trị J = {J.item():.2f}")
print(f"dJ/da = {a.grad.item():.2f}")
print(f"dJ/db = {b.grad.item():.2f}")
print(f"dJ/dc = {c.grad.item():.2f}")
print(f"dJ/dd = {d.grad.item():.2f}")