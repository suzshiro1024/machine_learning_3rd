import numpy as np

if __name__ == "__main__":
    ex = np.array([0, 1, 2, 3, 4, 5])

    standardized = (ex - ex.mean()) / ex.std()
    normalized = (ex - ex.min()) / (ex.max() - ex.min())

    print(f"standardized\n{standardized}")
    print(f"normalized\n{normalized}")
