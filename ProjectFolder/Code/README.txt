###############################################################################
# This code is the submission for the project AT4rAMT
#
# Andreas Streich, Heman Tanos, Malte Toetzke, Sebastian Windeck
# Deep Learning Lecture, Fall 2018
# Finalized January 19, 2019
###############################################################################


###############################################################################
# General comments & first steps:

To stat the code:
1. unzip the .zip file, and leave the structure as it is.
2. The MAPS database must be placed in the subfolder DeepLearningProject/ProjectFolder/Audiodaten (*.wav, *.mid and
   *.txt files). The MAPS data set is available from
   http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/
   Alternatively, you can place the extracted features in the subfolder DeepLearningProject/ProjectFolder/Features
3. Run the code by starting main.py in DeepLearningProject/ProjectFolder/Code


The parametrization is done via the dictionary args (starting in main.py on line 18). As it is at the moment, it will
generate the same experiments as used in the report, except:
- We did manually call take_every_second(args) after feature extraction, thus reducing the file size by a factor of 2
  in order to stay within the limit of the Leonhard cluster.
- Afther the training of the base model, we called take_every_second(args) as second time in order to have sufficient
  disk space to store intermediary results.

Furthermore, you can set parameters like the following ones in main.py
- train_basemodel (set to True for feature extraction, set to False to transcribe),
- noise_type ('simplistic' for uniform distribution, 'white', 'gaussian' or 'normal', 'pink', 'blue', 'brown', 'violet'),
- etc,

###############################################################################
# Specifically for running the code on the Leonhard cluster:
#
# 1. load the following required modules:
module --ignore-cache load "gcc/4.8.2"
module load "gcc/6.3.0"
module load "python_gpu/3.6.4"
module load hdf5/1.10.1
module load graphviz/2.40.1

# 2. install the required python packages
pip install -r requirements.txt

# To submit a job, do
bsub -Is -W 4:00 -R "rusage[mem=8192,ngpus_excl_p=1]"  python main.py

=> adjust the job length (here 4 hours, can also be 24:00) and the memory requirements (here 8 MB) to suit your needs.
=> To force usage of a GPU, add -R "select[gpu_model1==GeForceGTX1080Ti]" before "python" in the command line above.
