
import pandas
import matplotlib.pyplot as plt

FILE = "../data/Kazanina_gpt2-large_tokens_surprisal.txt"

df = pandas.read_csv(FILE, sep='\t', usecols=["condition", "surprisal"])

print(f"df:\n{df}")
df.info()

mean_df = df.pivot_table(index="condition")
sem_df = df.pivot_table(index="condition", aggfunc='sem')

print(f"mean df:\n{mean_df}")
print(f"sem df:\n{sem_df}")

mean_df.plot.bar(rot=0, yerr=sem_df, title="Kazanina et al 2007, surprisal")
plt.show()
