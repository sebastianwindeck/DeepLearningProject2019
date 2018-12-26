import csv
from config import create_config
from preprocess import preprocess, organize
from keras_train import train

# original:
#with open('models.csv', 'rb') as csvfile:
with open('models.csv', 'r') as csvfile:
     reader = csv.reader(csvfile)
     # Original:
     #header = reader.next()
     header = next(reader)
     print(header)
     for row in reader:
         args = dict(zip(header,row))
         print(args['model_name'])
         create_config(args)
         preprocess(args)
         organize(args)
         train(args)
