from category_gen import category_gen
import numpy as np

if __name__ == "__main__":
    df = category_gen()

    class_mapping = {
        label: idx for idx, label in enumerate(np.unique(df["classlabel"]))
    }
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    print(f"class+mapping\n{class_mapping}")
    print(f"inv_class_mapping\n{inv_class_mapping}")

    df["classlabel"] = df["classlabel"].map(class_mapping)
    print(f"df\n{df}")

    df["classlabel"] = df["classlabel"].map(inv_class_mapping)
    print(f"df\n{df}")
