
# import dependecies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline

df = pd.read_csv("HR_comma_sep")


CEO_dummies = pd.get_dummies(df.CEO, prefix="CEO_").astype("int")
salary_dummies = pd.get_dummies(df.salary, prefix="salary_").astype("int")

data = pd.concat([CEO_dummies, salary_dummies, df.drop("left", axis=1)], axis=1).drop(["CEO", "salary"], axis=1)
data = pd.concat([data, df.left], axis=1)


#PLOTS
# Correlation Heatmap
cor = data.corr()
mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(cor, mask=mask, vmax=.3, square=True)
plt.show()

g = sns.factorplot(x="satisfaction_level", y="promotion_last_5years",
                   hue="left", row="salary",
                    data=df,
                    orient="h", size=2, aspect=3.5, palette="Set3",
                    kind="violin", split=True, cut=0, bw=.2)
plt.show()


# data.to_csv("data_clean.csv", index=False, header=False)
