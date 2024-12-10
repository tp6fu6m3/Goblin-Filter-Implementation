from tqdm import tqdm

BASE_PATH = '../data/archive/'
data = []
with open(f'{BASE_PATH}/labels.txt', 'r', encoding='utf-8') as labels_file:
    labels = labels_file.readlines()
    for label in tqdm(labels):
        label = label.rstrip('\n').split(' ')[-1]
        data.append(float(label))

average = sum(data) / len(data)
print(average)

total = 0.
for x in data:
    total += (x-average)**2
MSE = total / len(data)
print(MSE)