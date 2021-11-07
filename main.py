# Import the necessary packages
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Acquisition
nfile = 2
seg_name = 'disf5'
path = ('C:/Users/felip/Desktop/Projetos/main-ms-felipe/files/database/')
train_path = (path+'training/')
test_path = (path+'tests/')

train_dir_name = os.listdir(train_path+seg_name)
test_dir_name = os.listdir(test_path+seg_name)

for i in test_dir_name:
    print("Processando a imagem", i)
    # Image reader
    img_name = test_path+seg_name+'/'+str(i)+'/'+str(i)+'_10_SEKELETON_WITHOUT_BRANCHS'+'.jpg'
    img = cv2.imread(img_name) # BRG order, uint8 type
    
    # Convert image to gray    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 0, 255, 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Save and read image in .pgm format
    img_name = test_path+seg_name+'/'+str(i)+'/'+str(i)+'_10_SEKELETON_WITHOUT_BRANCHS'+'.pgm'
    cv2.imwrite(img_name, gray)
    img = cv2.imread(test_path+seg_name+'/'+str(i)+'/'+str(i)+'_10_SEKELETON_WITHOUT_BRANCHS'+'.pgm')
    
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
    print(" O número total de pixels é", npixels)
    
    # Normalize Data and retrieval pixels with 0 or 255 values    
    #white = np.where(gray < [128], gray, [0])
    #black = np.where(gray > [128], gray, [255])
    
    # Retrieval white pixels coodinates from image
    white = np.where(img == [255])
    # Retrieval black pixels coodinates from image
    black = np.where(img == [0])
    
    # Zip (iterator) and convert to list for white pixels matrix (background)
    listzip1 = list(zip(white[0], white[1]))
    coord_bg = np.asarray(listzip1)
    #print(coord_bg)
    #print(len(coord_bg))
    
    # Zip (iterator) and convert to list for white pixels matrix (foreground)
    listzip2 = list(zip(black[0], black[1]))
    coord_fg = np.asarray(listzip2)
    #print(coord_fg)
    #print(len(coord_fg))
    
    # Amount of all seeds or coords number
    nseeds = int(len(coord_bg)+len(coord_fg))
    print("O número total de seeds/coordenadas é", nseeds)
    
    # Proportional seeds for black and white regions
    black_prop = len(coord_bg)/nseeds
    nblack_seeds = int(black_prop * nseeds)
    print("A proporção de sementes em regiões de úlcera é", round(float(black_prop)*100, 0))
    print("A quantidade de seeds preto é", nblack_seeds)
    
    white_prop = len(coord_fg)/nseeds
    nwhite_seeds = int(white_prop * nseeds)
    print("A proporção de sementes em regiões de não úlcera é", round(float(white_prop)*100, 0))
    print("A quantidade de seeds branco é", nwhite_seeds)
    
    # Setup the amount of rows from DataFrame (is the same of amount of seeds + header row)
    nrows = 8000
    print("A nova quantidade de seeds/coordenadas é", nrows-1)
    
    # New Proportional seeds for black and white regions
    nblack_seeds = int(black_prop * nrows)
    print("A proporção de sementes em regiões de úlcera é", round(float(black_prop)*100, 0))
    print("A quantidade de seeds preto é", nblack_seeds)
    
    nwhite_seeds = int(white_prop * nrows)
    print("A proporção de sementes em regiões de não úlcera é", round(float(white_prop)*100, 0))
    print("A quantidade de seeds branco é", nwhite_seeds)
    
    # Create a DataFrame
    #header = [nrows-1, width, height]
    df_header = pd.DataFrame({'x': [nrows], 'y': width, 'conn': height, 'label':''})
    df_white = pd.DataFrame({'x': coord_bg[:, 0], 'y': coord_bg[:, 1],'conn': 1,'label': 0})
    df_black = pd.DataFrame({'x': coord_fg[:, 0], 'y': coord_fg[:, 1],'conn': 1,'label': 1})
    #print(df_header)
    #print(df_white)
    #print(df_black)
    
    # Select rows randomly from DataFrame
    rdf_white = df_white.sample(n = nwhite_seeds, replace = True)
    rdf_black = df_black.sample(n = nblack_seeds, replace = True)
    
    if nfile == 1:
        # Concatenate DataFrame
        data = df_header.append(rdf_white)
        data = data.append(rdf_black)
        print(data)
    
        # Crete txt file to save the DataFrame content
        file = open(test_path+seg_name+'/'+str(i)+'/'+'seeds.txt', mode='w+')
        data.to_csv(test_path+seg_name+'/'+str(i)+'/'+'seeds.txt', sep='\t', index=False, header=False)
        file.close()
        print(">>> Um arquivo seeds.txt foi gravado com sucesso!")
    
    else:
        
        # Setup header with nseeds for white and black DataFrame
        wheader = [nwhite_seeds, width, height]
        df_wheader = pd.DataFrame({'x': [nwhite_seeds], 'y': width, 'conn': height, 'label':''})
        bheader = [nblack_seeds, width, height]
        df_bheader = pd.DataFrame({'x': [nblack_seeds], 'y': width, 'conn': height, 'label':''})
        
        # Concatenate DataFrame
        data = df_wheader.append(rdf_white)
        print(data)
        data2 = df_bheader.append(rdf_black)
        print(data2)
        
        # Crete txt file to save the DataFrame content
        file = open(test_path+seg_name+'/'+str(i)+'/'+'bg-seeds.txt', mode='w+')
        data.to_csv(test_path+seg_name+'/'+str(i)+'/'+'bg-seeds.txt', sep='\t', index=False, header=False)
        file.close()
        
        file = open(test_path+seg_name+'/'+str(i)+'/'+'ulcer-seeds.txt', mode='w+')
        data2.to_csv(test_path+seg_name+'/'+str(i)+'/'+'ulcer-seeds.txt', sep='\t', index=False, header=False)
        file.close()
        print(">>> Os arquivos ulcer-seeds.txt e bg-seeds.txt foram gravados com sucesso!")
         

print("Programa finalizado com sucesso!")
