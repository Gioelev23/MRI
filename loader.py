# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

import os
import tensorflow as tf
import numpy as np
import random
import nibabel as nib

#crea un flags indicando le dimensioni dell'immagine
FLAGS = tf.app.flags.FLAGS
FLAGS.width = 256
FLAGS.height = 320
FLAGS.depth = 320 
batch_index = 0
filenames = []

# prendi la cartella delle immagini
FLAGS.data_dir = "C:/Users/Gioele/Desktop/MRI/Images"
#numero di classi
FLAGS.num_class = 3


#funzione filenames che:
#- apre un file txt contenente i label separati da , e li inserisce in un vettore "labels"
#- per ciascuno di questi crea i filenames componendo filename e label
def get_filenames(data_set):
    global filenames
    labels = []
#apri il file txt,scorri ogni riga e splittala per virgole
    with open(FLAGS.data_dir + "/labels.txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list
            
#######################################################################
    
        list = os.listdir(FLAGS.data_dir  + '/' + data_set)
        i=0;
        for filename in list:
            filenames.append([filename, labels[i]])
            i+=1
#list prende i nomi dei file nella cartella data_set (train o test). Se ci aggiungo il
#label inteso come classe, lui metterà il label giusto appeso al nome del file in un vettore con
#l'indice i

    random.shuffle(filenames)
   
    
#######################################################################    
#funzione per il pick della data batch
def getdata(sess, data_set, batch_size):
    global batch_index, filenames
#preleva i filenames [filename, label]
    if len(filenames) == 0: get_filenames(data_set) 
    max = len(filenames)

#parte da batch_index e va avanti di batch_size (tanto ha già fatto shuffle randomico quindi l'ordine è sfalzato)
    begin = batch_index
    end = batch_index + batch_size
#se sfori finisci al massimo
    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, FLAGS.num_class)) # batch_sizeXclassi list for 'one hot encoding' 
    index = 0

#a questo punto cicli per batch size elementi
    for i in range(begin, end):
        
#prelevi il percorso del file giusto e carichi l'immagine nel modo giusto 
        imagePath = FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0]
        FA_org = nib.load(imagePath)
        
        FA_data = FA_org.get_data()  # 256x256x40; numpy.ndarray
        
        # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
        resized_image = tf.image.resize_images(images=FA_data, size=(FLAGS.width,FLAGS.height), method=1)
        #prende l'immagine e carica x (dati) e y (che sarebbe la colonna a cui mettere 1 e quindi il LABEL!)
        image = sess.run(resized_image)  # (256,256,40)
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        y_data[index][int(filenames[i][1])] = 1  # assign 1 to corresponding column (one hot encoding)
        index += 1
    #scorri avanti per la prossima batch e fai il reshape di x
    batch_index += batch_size  # update index for the next batch
    
    #x_data_ = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)
    x_data_ = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)
    return x_data_, y_data