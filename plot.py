import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from training_data import *


sb.set()


data = pd.read_csv('loss_logs.csv')

plt.plot('Iteration','Discriminator loss',data=data)
plt.plot('Iteration','Generator loss',data=data)

plt.legend()
plt.title('Training losses')
plt.tight_layout()

plt.savefig('plots/training_loss.png')
plt.show()