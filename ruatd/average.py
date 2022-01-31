import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

y_valid = pd.read_csv("/root/RuATD/data/binary/val.csv")["Class"].map({"H": 0, "M": 1})
model_files = [
    "mbart_ru_sum_gazeta",
    "mDeBERTa-v3-base-mnli-xnli",
    "rubert-base-cased",
    "xlm-roberta-large-en-ru",
]
target_cols = ["H", "M"]

df_valid = pd.concat(
    [
        pd.read_csv(f"/root/RuATD/submissions/binary/prob_valid_{i}.csv")[
            target_cols
        ].add_suffix(f"_{i}")
        for i in model_files
    ],
    axis=1,
)

df_test = pd.concat(
    [
        pd.read_csv(f"/root/RuATD/submissions/binary/prob_test_{i}.csv")[
            target_cols
        ].add_suffix(f"_{i}")
        for i in model_files
    ],
    axis=1,
)

clf = LogisticRegression(
    C=1, solver="newton-cg", penalty="l2", n_jobs=-1, max_iter=100
).fit(df_valid, y_valid)

print(clf.coef_)
y_test = clf.predict_proba(df_test)[:, 1]

submission = pd.read_csv("/root/RuATD/data/binary/test.csv")
submission["Class"] = (y_test >= np.median(y_test)).astype(int)
submission["Class"] = submission["Class"].map({0: "H", 1: "M"})

submission[["Id", "Class"]].to_csv(
    "/root/RuATD/submissions/binary/submission.csv", index=None
)
