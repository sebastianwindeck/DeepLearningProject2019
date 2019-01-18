import matplotlib.pyplot as plt
import os
import pandas

base_model_dir = 'ProjectFolder/Basemodel_1_pitch/'
base_model_file = base_model_dir + 'initialtraining.log'

noise_model_dir = 'ProjectFolder/Checkpoints/train2019-01-18-10-35-19/'

# load data on initial training:
tr_base = pandas.read_csv(base_model_file)
tr_base['phase'] = 'initial'

all_data = tr_base

fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
# load data on training with additional noise:
for s in os.listdir(noise_model_dir):
    if s[ -4:] == '.log':
        new_data = pandas.read_csv(noise_model_dir + s)
        # start: s[11:]
        numEnd = s[11:].find('t')
        thisEpoch = int(s[11:(11 + numEnd)])
        new_data['phase'] = thisEpoch
        all_data = all_data.append(new_data)
        ax[0][0].plot(new_data['epoch'], new_data['loss'])
        ax[1][0].plot(new_data['epoch'], new_data['f1'])
        ax[0][1].plot(new_data['epoch'], new_data['val_loss'])
        ax[1][1].plot(new_data['epoch'], new_data['val_f1'])

ax[0][0].set_ylabel('Training Loss')
ax[0][1].set_ylabel('Validation Loss')
ax[1][0].set_ylabel('Training F1')
ax[1][1].set_ylabel('Validation F1')
ax[1][0].set_xlabel('Epoch')
ax[1][1].set_xlabel('Epoch')
legend = fig.legend(['epoch ' + str(epo) for epo in range(thisEpoch+1)], loc='lower center', ncol=thisEpoch+1)



all_data['Epoch_overall']=range(all_data.shape[0])
fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
ax[0].plot(all_data['Epoch_overall'], all_data['loss'])
ax[0].plot(all_data['Epoch_overall'], all_data['val_loss'])
ax[0].set_ylabel('loss: Binary Cross-Entropy')
ax[0].legend()
ax[1].plot(all_data['Epoch_overall'], all_data['f1'])
ax[1].plot(all_data['Epoch_overall'], all_data['val_f1'])
ax[1].set_ylabel('F1 Score')
ax[1].set_xlabel('Epoch')
ax[1].legend()