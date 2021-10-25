# Import the necessary packages
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Acquisition
seg_name = 'disf5'
path = ('C:/Users/felip/Desktop/Projetos/main-ms-felipe/files/database/')
train_path = (path+'training/')
test_path = (path+'tests/')

train_dir_name = os.listdir(train_path+seg_name)
test_dir_name = os.listdir(test_path+seg_name)


for i in test_dir_name:
    print("Processando a imagem", i)
    # image reader
    img = cv2.imread(test_path+seg_name+'/'+str(i)+'/'+str(i)+'_10_SEKELETON_WITHOUT_BRANCHS'+'.jpg') # BRG order, uint8 type
    
    # Convert image to gray    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Show Image
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Get dimensions of image
    dim = img.shape
    # Height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    # Print dimensions
    print("As dimensões da imagem são", dim)
    npixels = height*width
    #print(" O número total de pixels é", npixels)
    
    # Retrieval white pixels coodinates from image
    white = np.where(gray == [255])
    # Retrieval black pixels coodinates from image
    black = np.where(gray == [0])
    
    # Zip (iterator) and convert to list for white pixels (background)
    listzip1 = list(zip(white[0], white[1]))
    coord_bg = np.asarray(listzip1)
    #print(coord_bg)
    #print(len(coord_bg))
    
    # Zip (iterator) and convert to list for white pixels (foreground)
    listzip2 = list(zip(black[0], black[1]))
    coord_fg = np.asarray(listzip2)
    #print(coord_fg)
    #print(len(coord_fg))
    
    # Seeds number
    nseeds = int(len(coord_bg)+len(coord_fg))
    print("O número total de seeds é", nseeds)
    
    # Create a DataFrame
    header = [nseeds, width, height]
    df_header = pd.DataFrame({'x': [nseeds], 'y': width, 'conn': height, 'label':''})
    df_white = pd.DataFrame({'x': coord_bg[:, 0], 'y': coord_bg[:, 1],'conn': 1,'label': 0})
    df_black = pd.DataFrame({'x': coord_fg[:, 0], 'y': coord_fg[:, 1],'conn': 1,'label': 1})
    #print(df_header)
    #print(df_white)
    #print(df_black)
    
    # Concatenate DataFrame
    data = df_header.append(df_white)
    data = data.append(df_black)
    print(data)
    
    # Crete txt file to save the DataFrame content
    file = open(test_path+seg_name+'/'+str(i)+'/'+'seeds.txt', mode='w+')
    data.to_csv(test_path+seg_name+'/'+str(i)+'/'+'seeds.txt', sep='\t', index=False, header=False)
    file.close()
    print(">>> Arquivo seeds.txt gravado com sucesso!")