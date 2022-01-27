import numpy as np
import pandas as pd

df1 = pd.read_csv("/root/RuATD/submissions/prob_mbart_ru_sum_gazeta.csv")
df2 = pd.read_csv("/root/RuATD/submissions/prob_rubert-base-cased.csv")
df3 = pd.read_csv("/root/RuATD/submissions/prob_xlm-roberta-large-en-ru.csv")

df1.Class = (df1.Class + df2.Class + df3.Class) / 3

df1.Class = df1.Class > np.median(df1.Class)
df1.Class = df1.Class.apply(lambda x: "H" if x else "M")

df1.to_csv("/root/RuATD/submissions/median_submission.csv", index=None)