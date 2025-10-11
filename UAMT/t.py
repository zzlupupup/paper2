import pickle

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)