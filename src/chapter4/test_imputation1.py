from nan_gen import nan_gen
from sklearn.impute import SimpleImputer
import numpy as np


if __name__ == "__main__":
    df = nan_gen()

    imr = SimpleImputer(missing_values=np.nan, strategy="mean")
    imr = imr.fit(df.values)

    imputed_data = imr.transform(df.values)
    print(f"imputed_data\n{imputed_data}")
