from micrograd import MLP, Value

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
ypred = []

mlp = MLP(3, [4, 4, 1])

loss: Value = Value(1)

for _ in range(1000):
    ypred = [mlp(x) for x in xs]
    loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)]) # type: ignore
    print([y.data for y in ypred] , loss.data) # type: ignore

    for p in mlp.parameters():
        p.grad = 0
    loss.backward()
    
    for p in mlp.parameters():
        p.data += -0.1 * p.grad
else:
    print(loss.data)
    print(mlp.parameters())
    print(ypred)
