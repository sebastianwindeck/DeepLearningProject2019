import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas
import numpy as np

base_model_dir = '../../ProjectFolder/Basemodel_1_pitch/'
base_model_file = base_model_dir + 'initialtraining.log'

noise_model_dir = '../../ProjectFolder/Checkpoints/train2019-01-18-10-35-19/'

# load data on initial training:
tr_base = pandas.read_csv(base_model_file)
tr_base['phase'] = 'initial'

all_data = tr_base

input_level = np.load(noise_model_dir + 'input_level.npy')
noise = np.load(noise_model_dir + 'noise_levels.npy')
bm_score = np.load(noise_model_dir + 'bm_score.npy')
bm_score= np.reshape(bm_score, newshape=(-1,2))

# Figure 1 Training and Validation uising Loss and Metric function over epochs
colors = cm.rainbow(np.linspace(0, 1, 15))
fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, sharex=False)
# load data on training with additional noise:
noise_f1 = []
noise_loss = []
i = 0
for s in os.listdir(noise_model_dir):
    if s[-4:] == '.log':
        new_data = pandas.read_csv(noise_model_dir + s)
        # start: s[11:]
        numEnd = s[11:].find('t')
        thisEpoch = int(s[11:(11 + numEnd)])
        print(thisEpoch)
        new_data['phase'] = thisEpoch
        all_data = all_data.append(new_data)
        ax[0][0].plot(new_data['epoch'], new_data['loss'], alpha=1, color=colors[i])
        ax[1][0].plot(new_data['epoch'], new_data['f1'], alpha=1, color=colors[i])
        ax[0][1].plot(new_data['epoch'], new_data['val_loss'], alpha=1, color=colors[i])
        ax[1][1].plot(new_data['epoch'], new_data['val_f1'], alpha=1, color=colors[i])
        noise_f1.append(max(new_data['val_f1']))
        noise_loss.append(min(new_data['val_loss']))
        i += 1

ax[0][0].set_ylabel('Training Loss')
ax[0][1].set_ylabel('Validation Loss')
ax[1][0].set_ylabel('Training F1')
ax[1][1].set_ylabel('Validation F1')
ax[1][1].set_xlabel('epoch')
ax[1][0].set_xlabel('epoch')
ax[0][0].tick_params(labelsize='small')
ax[0][1].tick_params(labelsize='small')
ax[1][0].tick_params(labelsize='small')
ax[1][1].tick_params(labelsize='small')
fig.suptitle('Training epochs on noisy data with metric and loss')

fig.subplots_adjust(bottom=0.3, wspace=0.33)

fig.legend(['epoch ' + str(epo) for epo in range(thisEpoch + 1)], ncol=int((thisEpoch+1)/2), fontsize ='xx-small', loc="center")
plt.tight_layout(pad=10, w_pad=0, h_pad=4)

plt.show()

# Learning curve of training and validation set
all_data['Epoch_overall'] = range(all_data.shape[0])
fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')

ax[0].plot(all_data['Epoch_overall'], all_data['loss'])
ax[0].plot(all_data['Epoch_overall'], all_data['val_loss'])
ax[0].set_ylabel('Weighted Binary Cross-Entropy')
ax[0].legend(loc='lower right')
ax[1].plot(all_data['Epoch_overall'], all_data['f1'])
ax[1].plot(all_data['Epoch_overall'], all_data['val_f1'])
ax[1].set_ylabel('F1 Score')
ax[1].set_xlabel('Epoch')
ax[1].legend(loc='lower right')
fig.suptitle('Learning curve with metric and loss function')

plt.tight_layout(pad=5, w_pad=0, h_pad=1.0)
plt.show()

# Noise development curve
fig = plt.Figure()
plt.plot(noise, label='noise level')
plt.hlines(y=input_level, xmin=0, xmax=noise.shape[0]-1, color='r', linestyles='--', label='average sound level')
plt.title('Noise level intensity')
plt.xticks(np.arange(noise.shape[0]))
plt.xlabel('epoch')
plt.ylabel('noise')
plt.legend()
plt.show()

# Base model development curve
fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
ax[0].plot(bm_score[:,0], label='Clear loss', color='r', alpha=0.5)
ax[1].plot(bm_score[:,1], label='Clear f1-score', color='r', alpha=1)
ax[0].plot(noise_loss, label='Noisy loss', color='g', alpha=0.5)
ax[1].plot(noise_f1, label='Noisy f1-score', color='g', alpha=1)
ax[0].set_title('Clear vs. noisy score for noise levels')
ax[0].set_xticks(np.arange(bm_score.shape[0]))
ax[1].set_xticks(np.arange(bm_score.shape[0]))
ax[1].set_xlabel('Noise epoch')
ax[1].set_ylabel('F1 Score')
ax[0].set_ylabel('Weighted Binary Cross-Entropy')
ax[0].legend()
ax[1].legend()
plt.show()
