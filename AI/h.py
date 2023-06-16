import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 15, 25]

plt.plot(x, y)

ticks = [1, 2, 3, 4]
labels = ['A', 'B', 'C', 'D']

plt.xticks(ticks, labels)

plt.show()


