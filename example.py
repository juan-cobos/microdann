import random 
from microdann import DANN

random.seed(43)

xs = [3, 5, 6]

dann = DANN(3, 2, 2, 2)
ypred = dann(xs)
y = [1, 0]

loss = sum([(yi - ygt)**2 for yi, ygt in zip(ypred, y)])
loss.backward()

print(len([p for p in dann.parameters() if p.grad!=0]))