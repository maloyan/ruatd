import string

import catboost as cb
import pandas as pd
from catboost import CatBoostClassifier, Pool


def get_len(df):
    df["Length"] = df["Text"].str.len()

    count = lambda l1,l2: sum([1 for x in l1 if x in l2])

    df["Punctuation"] = count(df["Text"],set(string.punctuation)) 
    return df


df_train = pd.read_csv("/root/RuATD/data/binary/train.csv")
df_valid = pd.read_csv("/root/RuATD/data/binary/val.csv")
df_test = pd.read_csv("/root/RuATD/data/binary/test.csv")

df_train.Class = df_train.Class.apply(lambda x: 1 if x == "M" else 0)
df_valid.Class = df_valid.Class.apply(lambda x: 1 if x == "M" else 0)

df_train = get_len(df_train)
df_valid = get_len(df_valid)
df_test = get_len(df_test)

y_train = df_train.pop("Class")
y_valid = df_valid.pop("Class")

train_pool = Pool(
    df_train[["Text", "Length", "Punctuation"]],
    y_train,
    text_features=["Text"],
    feature_names=list(["Text", "Length", "Punctuation"]),
)
valid_pool = Pool(
    df_valid[["Text", "Length", "Punctuation"]],
    y_valid,
    text_features=["Text"],
    feature_names=list(["Text", "Length", "Punctuation"]),
)
test_pool = Pool(
    df_test[["Text", "Length", "Punctuation"]],
    text_features=["Text"],
    feature_names=list(["Text", "Length", "Punctuation"]),
)
catboost_params = {
    "eval_metric": "Accuracy",
    "task_type": "GPU",
    "iterations": 10000,
    "use_best_model": True,
    "verbose": 100,
}
model = CatBoostClassifier(**catboost_params)
model.fit(train_pool, eval_set=valid_pool)
df_test["Class"] = model.predict(test_pool)
df_test["Class"] = df_test["Class"].apply(lambda x: "M" if x else "H")

df_test[["Id", "Class"]].to_csv("/root/RuATD/submissions/cb.csv", index=None)
