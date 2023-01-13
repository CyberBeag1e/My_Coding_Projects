import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import glob
import random

from keras import Model
from keras.layers import Layer,add,\
                         Activation,BatchNormalization,Conv2D,MaxPooling2D,\
                         GlobalAveragePooling2D,Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.models import Sequential,load_model

from sklearn.model_selection import train_test_split

"""
The objective of this coding project is to detect a pattern in a picture.
The model used is ResNet18 / ResNet34, as defined in part I
The image processing and detecting model is defined in part II
"""

## -------------------------------------------------------------------
##### Part I #####
## Define the basic block in ResNet model, which contains:
##      1. A 'same' convolution layer, with kernel size 3*3
##      2. A batch normalization layer
##      3. An activation layer, with 'relu' function
##      4. Another 'same' convolution layer, with kernel size 3*3
##      5. A batch normalization layer
## -------------------------------------------------------------------

class ResBlock(Layer):

    def __init__(self,kernelNum,strides=1):
        super().__init__()

        self.conv1=Conv2D(kernelNum,[3,3],padding="same",strides=strides)
        self.btn1=BatchNormalization()
        self.relu=Activation('relu')

        self.conv2=Conv2D(kernelNum,[3,3],padding="same",strides=1)
        self.btn2=BatchNormalization()

        ## downsample the residual
        """
        In some of the ResNet blocks, we need to double the depth of the tensor,
        as well as half the tensor's size.
        We add a self-defined `strides` parameter to control the downsampling action
        """
        if strides != 1:
            self.downsample = Sequential([
                Conv2D(kernelNum,[3,3],strides=strides,padding='same'),
                BatchNormalization(),
                Activation('relu')
            ])
        else:
            ## When `strides` is set to default (1), we don't need downsampling
            self.downsample = lambda x:x
    
    ## Put together the layers
    def call(self,inputs):
        x=self.conv1(inputs)
        x=self.btn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.btn2(x)

        ## downsample the residual 
        x0=self.downsample(inputs)
        x=add([x,x0])
        x=self.relu(x)

        return x

## Define the ResNet model
class ResNet(Model):
    def __init__(self,dims,classNum=100):
        super().__init__()

        ## Data 'preprocessing'
        self.stem=Sequential([
            Conv2D(64,[3,3],padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D([3,3],padding='same',strides=[1,1])
        ])

        ## Add the ResBlock
        ## We don't need downsampling the residual in the first block
        self.layer1=self.addResBlock(64,dims[0])

        self.layer2=self.addResBlock(128,dims[1],strides=2)
        self.layer3=self.addResBlock(256,dims[2],strides=2)
        self.layer4=self.addResBlock(512,dims[3],strides=2)

        self.avgPool=GlobalAveragePooling2D()
        ## Add Dense layer and activate it by `softmax` function for classification
        self.dense=Dense(classNum,activation='softmax')

    def call(self,inputs):
        x=self.stem(inputs)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgPool(x)
        ## I attempt to use the setting of Dense layers in LeNet5 Model
        x=Dense(120,activation='relu')(x)
        x=Dense(84,activation='relu')
        output=self.dense(x)

        return output
    
    ## Add basic ResNet blocks
    def addResBlock(self,kernelNum,blocks,strides=1):
        resBlocks=Sequential()
        """
        In the first sub-block, we need to downsample the residual, 
        so we use the self-defined `strides` parameter
        """
        resBlocks.add(ResBlock(kernelNum,strides))

        for _ in range(1,blocks):
            resBlocks.add(ResBlock(kernelNum,strides=1))
        
        return resBlocks


def ResNetModel(classNum,version="18"):
    if version=="18":           ## ResNet18
        return ResNet([2,2,2,2],classNum)
    elif version=="34":         ## ResNet34
        return ResNet([3,4,6,3],classNum)
    else:
        return

    

## -------------------------------------------------------------------
##### Part II #####
## Define Pattern Detection Model
## -------------------------------------------------------------------

class PatternDetection:

    def __init__(self):
        self.patterns=[]
        self.backgrounds=None

        self.sampleNum=None
        self.ptsize=None
        self.bgsize=None

        self.model=None
        self.bestModel=None

    ## Load pattern images and convert to np.ndarray
    def addPatterns(self,img_path,ptSize=128):
        self.ptsize=ptSize
        dirLs=glob.glob(os.path.join(img_path,"*.jpg"))
        for d in dirLs:
            img=Image.open(d).resize([ptSize,ptSize])
            imgArr=np.array(img)/255.
            self.patterns.append(imgArr)
        print(f"{len(self.patterns)} patterns added.")

    ## Load background images and convert to np.ndarray
    def addBackGrounds(self,bgs_path,resized_shape=256):
        self.bgsize=resized_shape
        dirLs=glob.glob(os.path.join(bgs_path,"*.jpg"))
        np.random.shuffle(dirLs)
        self.sampleNum=len(dirLs)
        self.backgrounds=np.zeros([self.sampleNum,self.bgsize,self.bgsize,3])

        for i in range(self.sampleNum):
            bg=Image.open(dirLs[i]).resize([resized_shape,resized_shape]).convert('RGB')
            bgArr=np.array(bg)/255.
            self.backgrounds[i,]=bgArr

        print(f"{self.sampleNum} background images added.")


    def printPattern(self,bgImg):
        """
        To obtain ample data to train our model:
            1. I selected several pattern pictures and thousands of background images
            2. I 'print` the pattern onto the background, by using a pattern array
               to 'multiply' a parts of the background array
            3. Hence, I generated pictures of the pattern in multiple backgrounds
        """
        pattern=random.choice(self.patterns)
        ptSize=self.ptsize

        bgShape=bgImg.shape[0]
        ## Make sure that the pattern is 'printed' within the background boundary
        x0=np.random.randint(0,bgShape-ptSize)
        y0=np.random.randint(0,bgShape-ptSize)
        x1=x0+ptSize
        y1=y0+ptSize

        ## Print the pattern.
        ## Multiplying ensures that the range of image array is within (0,1)
        bgImg[x0:x1,y0:y1,:]=bgImg[x0:x1,y0:y1,:]*pattern

        return bgImg

    ## Image cropping
    def imgCrop(self):
        """
        We hope to train our model to detect both full pattern and part of the pattern,
        so we crop the image, leaving the 'middle' of the picture to ensure that at least 
        one part of the pattern is presented on the picture
        """
        x0=int(0.5*self.ptsize);x1=self.bgsize-int(0.5*self.ptsize)
        y0=int(0.5*self.ptsize);y1=self.bgsize-int(0.5*self.ptsize)
        return x0,x1,y0,y1
    

    ## Prepare data for model training
    def dataPrepare(self,proportion=0.5):
        """
        I need to label the pictures for classification
        Therefore, pictures should firstly be divided into one group with patterns 
        and the other group without patterns
        """
        X=self.backgrounds
        ## Number of pictures with patterns on them
        printed_num=round(self.sampleNum*proportion)

        ## One-hot encoding for Y
        Y=np.zeros([self.sampleNum,2])
        Y[:printed_num,0]=0;Y[:printed_num,1]=1
        Y[printed_num:,0]=1;Y[printed_num:,1]=0

        ## Print patterns
        for i in range(printed_num):
            X[i,...]=self.printPattern(X[i,...])

        ## Crop images   
        x_0,x_1,y_0,y_1=self.imgCrop()
        X=X[:,x_0:x_1,y_0:y_1,:]
        
        return X,Y
    
    ## Add Model
    def addModel(self,model):
        self.model=model
    
    ## Train Model
    def modelTraining(self,optimizer,lr_scheduler):
        ## Data preparation
        X,Y=self.dataPrepare()
        X0,X1,Y0,Y1=train_test_split(X,Y,test_size=0.3,random_state=666)
        trainGenerator=imgGenerator(X0,Y0)

        ## Build the model
        input_shape=(None,X.shape[1],X.shape[2],X.shape[3])
        self.model.build(input_shape=input_shape)
        self.model.summary()

        ## Compile our model
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=optimizer,
                           metrics=['accuracy'])
        
        callbacksLs=[lr_scheduler]

        ## Use model check points to save the best model
        ## 'Best' is defined as the highest accuracy on validation data
        callback=ModelCheckpoint(filepath="checkpoints",
                                 monitor='val_accuracy',
                                 mode='max',
                                 save_best_only=True,
                                 verbose=0)

        callbacksLs.append(callback)

        ## Fit model
        fitted=self.model.fit(trainGenerator,
                              epochs=50,
                              validation_data=(X1,Y1),
                              verbose=0,
                              callbacks=callbacksLs)
        
        self.bestModel=load_model("checkpoints")

        return fitted
        
    ## Pattern detection
    ## Use simulation, to generate the 'sub-image' of the new picture
    def detect(self,newImg,num_of_sims=1000):
        nx,ny,nz=newImg.shape
        ## Save coordinates of the 'sub-images'
        locs=np.zeros([num_of_sims,2])
        ## Save 'sub-images'
        subs=np.zeros([num_of_sims,self.ptsize,self.ptsize,3])

        for i in range(num_of_sims):
            x0=np.random.randint(0,nx-self.ptsize)
            x1=x0+self.ptsize
            y0=np.random.randint(0,ny-self.ptsize)
            y1=y0+self.ptsize

            ## Randomly get the sub-image from the new picture
            subs[i]=newImg[x0:x1,y0:y1,:]
            locs[i]=[x0,y0]
        
        ## Use fitted model to predict the probabilities that the sub-image includes the pattern
        probs=self.bestModel.predict(subs)
        ## Get the sub-image that is most likely to have a pattern on
        idx0,idx1=np.argmax(probs,axis=0)
        sub_x,sub_y=locs[idx1]
        sub_x=int(sub_x)
        sub_y=int(sub_y)

        ## Change values of one of the RGB channels, to highlight the pattern on the picture
        newImg[sub_x:(sub_x+self.ptsize),sub_y:(sub_y+self.ptsize),1]=0
        plt.imshow(newImg)



## Define the image data generator to perform data augmentation
def imgGenerator(x,y,batchSize=500):

    img_generator=ImageDataGenerator(
        shear_range=0.5,
        horizontal_flip=True)\
        .flow(
            x,y,
            batch_size=batchSize
        )
    
    return img_generator


def main():
    img_path="../Files/patterns"
    bgs_path="../Files/backgrounds"

    detectionModel=PatternDetection()
    detectionModel.addPatterns(img_path)
    detectionModel.addBackGrounds(bgs_path)

    model=ResNetModel(classNum=2,version="18")
    detectionModel.addModel(model)

    ## Define the learning rate scheduler with decay
    def scheduler(epoch,init_lr=0.001,lr_decay=10):
        lr=init_lr*(0.05**(epoch//lr_decay))
        return lr
    
    lr_scheduler=LearningRateScheduler(scheduler,verbose=0)
    optimizer=Adam(0.001)

    detectionModel.modelTraining(optimizer=optimizer,lr_scheduler=lr_scheduler)

    new_path="<new img path>"
    img=Image.open(new_path).resize([480,480])
    imgArr=np.array(img)/255.
    detectionModel.detect(imgArr)