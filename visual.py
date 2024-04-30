import seaborn as sns
import numpy as np 
import torch
from sklearn.manifold import TSNE
from collections import Counter
import os
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
sns.set_style("white")
cnt = 6
palette = sns.color_palette("bright", cnt)


# (22144, 189, 63)
epoch = 118


t = 'private'
base_path = f'./visual_ts/{t}/{epoch}'
pri_f = [i for i in os.listdir(base_path) if i.endswith('.pt')]
pri_ts = []
pri_lb = []
for i in pri_f:
    pri_i = dict(torch.load(os.path.join(base_path, i)).items())
    pri_ts.append(pri_i['data'])
    pri_lb.append(pri_i['label'])
pri_samples_np = np.concatenate(pri_ts, axis=0)
pri_label_np = np.concatenate(pri_lb, axis=0)

pri_ind = (pri_label_np == 0) | (pri_label_np == 1) | (pri_label_np == 2) | (pri_label_np == 3) | (pri_label_np == 4) | (pri_label_np == 5)
pri_samples_np = pri_samples_np[pri_ind]
pri_label_np = pri_label_np[pri_ind]

# X.view(128, -1, 63, 63).
z = Counter(list(pri_label_np))
print(sorted(z.items(), key=lambda x: x[1]))


pri_label_np = np.array(["label" + str(x) for x in list(pri_label_np)])
print(pri_label_np)
print(pri_samples_np.shape, pri_label_np.shape)



tsne = TSNE(n_components = 2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(pri_samples_np)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# print(X_norm.shape)
fig = sns.scatterplot(X_norm[:,0], X_norm[:,1], hue=pri_label_np, legend='full', palette=palette)
fig_obj = fig.get_figure()
fig_obj.savefig(os.path.join(base_path, 'out1.png'), dpi = 400)