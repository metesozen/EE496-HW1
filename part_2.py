#Mete Sözen
#2031375
import numpy as np

from utils import part2Plots

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

train_images=np.load('train_images.npy')#load dataset train_images_shape=(30000,784) by print(train_images.shape)
train_labels=np.load('train_labels.npy')#(30000,)
test_images=np.load('test_images.npy')#(5000,784)
test_labels=np.load('test_labels.npy')#(5000,)


train_images = ((train_images-127.5)/127.5)#preprocess images pixel values between -1 and 1
test_images=((test_images-127.5)/127.5)

X_train, X_validate, y_train, y_validate = train_test_split(train_images, train_labels, test_size=0.1,)#split training and validation data
class_names = [0,1,2,3,4]#y_train vary between 0 and 4

arch1 = MLPClassifier(hidden_layer_sizes=(128,), batch_size=500, verbose=True, max_iter=1, n_iter_no_change=5400, warm_start=True)
arch2 = MLPClassifier(hidden_layer_sizes=(16, 128),batch_size=500, verbose=True)
arch3 = MLPClassifier(hidden_layer_sizes=(16, 128, 16),batch_size=500, verbose=True)
arch5 = MLPClassifier(hidden_layer_sizes=(16, 128, 64, 32, 16),batch_size=500, verbose=True)
arch7 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16),batch_size=500, verbose=True)#verbose true to see each iteration

#27000 samples, 500 batch size.
#1 epoch=54 iterations (27000/500)
#100 epoch=5400 iterations
#partial_fit method updates the model with a single iteration over the given data

#run architecture 1 at least 10 times and calculate average loss and accuracy curves:
k=0
n = 10
traininglossarch1 = [[] for _ in range(n)]#create 10 arrays to take their average after 10 times running each architecture.
trainingaccuracyarch1=[[] for _ in range(n)]
validationaccuracyarch1=[[] for _ in range(n)]
testaccuracyarch1=[]
weightsofthefirsthiddenlayerarch1=[]
for t in range(10):
    for j in range (5400):
        if k==0:
            arch1=arch1.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step
            k=k+1
        else:
            arch1=arch1.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step, no need to pass class argument after 1st iteration
            k=k+1
        if (k%10==0):#at every 10 step, save loss and accuracies
            trainingaccuracyarch1item=arch1.score(X_train, y_train, sample_weight=None)#take training accuracy score 
            trainingaccuracyarch1[t].append(trainingaccuracyarch1item)
            validationaccuracyarch1item=arch1.score(X_validate,y_validate)#take validation accuracy score
            validationaccuracyarch1[t].append(validationaccuracyarch1item)
            traininglossarch1item=arch1.loss_#take training loss 
            traininglossarch1[t].append(traininglossarch1item)
            idx = np.random.permutation(len(X_train))
            X_train,y_train = X_train[idx], y_train[idx]#shuffle at each 10th approach (iteration, step)
    print("done")
    k=0
    weightsofthefirsthiddenlayerarch1item=arch1.coefs_[0]
    weightsofthefirsthiddenlayerarch1.append(weightsofthefirsthiddenlayerarch1item)
    testaccuracyarch1item=arch1.score(test_images,test_labels)
    testaccuracyarch1.append(testaccuracyarch1item)
    arch1 = MLPClassifier(hidden_layer_sizes=(128,), batch_size=500, verbose=True, max_iter=1, n_iter_no_change=5400, warm_start=True)#initialize the architecture

traininglossarch1avg = []#average curves
trainingaccuracyarch1avg=[]
validationaccuracyarch1avg=[]
for j in range (5400):#calculating average validation, training accuracy and training loss
    for t in range (10):
        traininglossarch1sum=traininglossarch1sum+traininglossarch1[t][j]
        trainingaccuracyarch1sum=trainingaccuracyarch1sum+trainingaccuracyarch1[t][j]
        validationaccuracyarch1sum=validationaccuracyarch1sum+validationaccuracyarch1[t][j]
    traininglossarch1avg.append(traininglossarch1sum/10.0)
    trainingaccuracyarch1avg.append(trainingaccuracyarch1sum/10.0)
    validationaccuracyarch1avg.append(validationaccuracyarch1sum/10.0)

for t in range (10):

    testaccuracyarch1sum=testaccuracyarch1sum+testaccuracyarch1[t]
    weightsofthefirsthiddenlayerarch1sum=weightsofthefirsthiddenlayerarch1sum+weightsofthefirsthiddenlayerarch1[t]
testaccuracyarch1avg=testaccuracyarch1sum/10.0
weightsofthefirsthiddenlayerarch1avg=weightsofthefirsthiddenlayerarch1sum/10.0


#run architecture 2 at least 10 times and calculate average loss and accuracy curves:

k=0
n = 10
traininglossarch2 = [[] for _ in range(n)]#create 10 arrays to take their average after 10 times running each architecture.
trainingaccuracyarch2=[[] for _ in range(n)]
validationaccuracyarch2=[[] for _ in range(n)]
testaccuracyarch2=[]
weightsofthefirsthiddenlayerarch2=[]
for t in range(10):
    for j in range (5400):
        if k==0:
            arch2.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step
            k=k+1
        else:
            arch2.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step, no need to pass class argument after 1st iteration
            k=k+1
        if (k%10==0):#at every 10 step, save loss and accuracies
            trainingaccuracyarch2item=arch2.score(X_train, y_train, sample_weight=None)#take training accuracy score 
            trainingaccuracyarch2[t].append(trainingaccuracyarch1item)
            validationaccuracyarch2item=arch2.score(X_validate,y_validate)#take validation accuracy score
            validationaccuracyarch2[t].append(validationaccuracyarch1item)
            traininglossarch2item=arch2.loss_#take training loss 
            traininglossarch2[t].append(traininglossarch1item)
            idx = np.random.permutation(len(X_train))
            X_train,y_train = X_train[idx], y_train[idx]#shuffle at each 10th approach (iteration, step)
    print("done")
    k=0
    weightsofthefirsthiddenlayerarch2item=arch2.coefs_[0]
    weightsofthefirsthiddenlayerarch2.append(weightsofthefirsthiddenlayerarch2item)
    testaccuracyarch2item=arch2.score(test_images,test_labels)
    testaccuracyarch2.append(testaccuracyarch2item)
    arch2 = MLPClassifier(hidden_layer_sizes=(16, 128),batch_size=500, verbose=True)#initialise arch2


traininglossarch2avg = []#average curves
trainingaccuracyarch2avg=[]
validationaccuracyarch2avg=[]
for j in range (5400):#calculating average validation, training accuracy and training loss
    for t in range (10):
        traininglossarch2sum=traininglossarch2sum+traininglossarch2[t][j]
        trainingaccuracyarch2sum=trainingaccuracyarch2sum+trainingaccuracyarch2[t][j]
        validationaccuracyarch2sum=validationaccuracyarch2sum+validationaccuracyarch2[t][j]
    traininglossarch2avg.append(traininglossarch2sum/10.0)
    trainingaccuracyarch2avg.append(trainingaccuracyarch2sum/10.0)
    validationaccuracyarch2avg.append(validationaccuracyarch2sum/10.0)

for t in range (10):

    testaccuracyarch2sum=testaccuracyarch2sum+testaccuracyarch2[t]
    weightsofthefirsthiddenlayerarch2sum=weightsofthefirsthiddenlayerarch2sum+weightsofthefirsthiddenlayerarch2[t]
testaccuracyarch2avg=testaccuracyarch2sum/10.0
weightsofthefirsthiddenlayerarch2avg=weightsofthefirsthiddenlayerarch2sum/10.0

#run architecture 3 at least 10 times and calculate average loss and accuracy curves:
k=0
n = 10
traininglossarch3 = [[] for _ in range(n)]#create 10 arrays to take their average after 10 times running each architecture.
trainingaccuracyarch3=[[] for _ in range(n)]
validationaccuracyarch3=[[] for _ in range(n)]
testaccuracyarch3=[]
weightsofthefirsthiddenlayerarch3=[]
for t in range(10):
    for j in range (5400):
        if k==0:
            arch3.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step
            k=k+1
        else:
            arch3.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step, no need to pass class argument after 1st iteration
            k=k+1
        if (k%10==0):#at every 10 step, save loss and accuracies
            trainingaccuracyarch3item=arch3.score(X_train, y_train, sample_weight=None)#take training accuracy score 
            trainingaccuracyarch3[t].append(trainingaccuracyarch1item)
            validationaccuracyarch3item=arch3.score(X_validate,y_validate)#take validation accuracy score
            validationaccuracyarch3[t].append(validationaccuracyarch1item)
            traininglossarch3item=arch3.loss_#take training loss 
            traininglossarch3[t].append(traininglossarch1item)
            idx = np.random.permutation(len(X_train))
            X_train,y_train = X_train[idx], y_train[idx]#shuffle at each 10th approach (iteration, step)
    print("done")
    k=0
    weightsofthefirsthiddenlayerarch3item=arch3.coefs_[0]
    weightsofthefirsthiddenlayerarch3.append(weightsofthefirsthiddenlayerarch3item)
    testaccuracyarch3item=arch3.score(test_images,test_labels)
    testaccuracyarch3.append(testaccuracyarch3item)
    arch3 = MLPClassifier(hidden_layer_sizes=(16, 128, 16),batch_size=500, verbose=True)#initialise arch 3



traininglossarch3avg = []#average curves
trainingaccuracyarch3avg=[]
validationaccuracyarch3avg=[]
for j in range (5400):#calculating average validation, training accuracy and training loss
    for t in range (10):
        traininglossarch3sum=traininglossarch3sum+traininglossarch3[t][j]
        trainingaccuracyarch3sum=trainingaccuracyarch3sum+trainingaccuracyarch3[t][j]
        validationaccuracyarch3sum=validationaccuracyarch3sum+validationaccuracyarch3[t][j]
    traininglossarch3avg.append(traininglossarch3sum/10.0)
    trainingaccuracyarch3avg.append(trainingaccuracyarch3sum/10.0)
    validationaccuracyarch3avg.append(validationaccuracyarch3sum/10.0)

for t in range (10):

    testaccuracyarch3sum=testaccuracyarch3sum+testaccuracyarch3[t]
    weightsofthefirsthiddenlayerarch3sum=weightsofthefirsthiddenlayerarch3sum+weightsofthefirsthiddenlayerarch3[t]
testaccuracyarch3avg=testaccuracyarch3sum/10.0
weightsofthefirsthiddenlayerarch3avg=weightsofthefirsthiddenlayerarch3sum/10.0

#run architecture 5 at least 10 times and calculate average loss and accuracy curves:
k=0
n = 10
traininglossarch5 = [[] for _ in range(n)]#create 10 arrays to take their average after 10 times running each architecture.
trainingaccuracyarch5=[[] for _ in range(n)]
validationaccuracyarch5=[[] for _ in range(n)]
testaccuracyarch5=[]
weightsofthefirsthiddenlayerarch5=[]
for t in range(10):
    for j in range (5400):
        if k==0:
            arch5.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step
            k=k+1
        else:
            arch5.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step, no need to pass class argument after 1st iteration
            k=k+1
        if (k%10==0):#at every 10 step, save loss and accuracies
            trainingaccuracyarch5item=arch5.score(X_train, y_train, sample_weight=None)#take training accuracy score 
            trainingaccuracyarch5[t].append(trainingaccuracyarch1item)
            validationaccuracyarch5item=arch5.score(X_validate,y_validate)#take validation accuracy score
            validationaccuracyarch5[t].append(validationaccuracyarch1item)
            traininglossarch5item=arch5.loss_#take training loss 
            traininglossarch5[t].append(traininglossarch1item)
            idx = np.random.permutation(len(X_train))
            X_train,y_train = X_train[idx], y_train[idx]#shuffle at each 10th approach (iteration, step)
    print("done")
    k=0
    weightsofthefirsthiddenlayerarch5item=arch5.coefs_[0]
    weightsofthefirsthiddenlayerarch5.append(weightsofthefirsthiddenlayerarch3item)
    testaccuracyarch5item=arch3.score(test_images,test_labels)
    testaccuracyarch5.append(testaccuracyarch3item)
    arch5 = MLPClassifier(hidden_layer_sizes=(16, 128, 64, 32, 16),batch_size=500, verbose=True)#initialise arch5



traininglossarch5avg = []#average curves
trainingaccuracyarch5avg=[]
validationaccuracyarch5avg=[]
for j in range (5400):#calculating average validation, training accuracy and training loss
    for t in range (10):
        traininglossarch5sum=traininglossarch5sum+traininglossarch5[t][j]
        trainingaccuracyarch5sum=trainingaccuracyarch5sum+trainingaccuracyarch5[t][j]
        validationaccuracyarch5sum=validationaccuracyarch5sum+validationaccuracyarch5[t][j]
    traininglossarch5avg.append(traininglossarch5sum/10.0)
    trainingaccuracyarch5avg.append(trainingaccuracyarch5sum/10.0)
    validationaccuracyarch5avg.append(validationaccuracyarch5sum/10.0)

for t in range (10):

    testaccuracyarch5sum=testaccuracyarch5sum+testaccuracyarch5[t]
    weightsofthefirsthiddenlayerarch5sum=weightsofthefirsthiddenlayerarch5sum+weightsofthefirsthiddenlayerarch5[t]
testaccuracyarch5avg=testaccuracyarch5sum/10.0
weightsofthefirsthiddenlayerarch5avg=weightsofthefirsthiddenlayerarch5sum/10.0


#run architecture 7 at least 10 times and calculate average loss and accuracy curves:
k=0
n = 10
traininglossarch7 = [[] for _ in range(n)]#create 10 arrays to take their average after 10 times running each architecture.
trainingaccuracyarch7=[[] for _ in range(n)]
validationaccuracyarch7=[[] for _ in range(n)]
testaccuracyarch7=[]
weightsofthefirsthiddenlayerarch7=[]
for t in range(10):
    for j in range (5400):
        if k==0:
            arch7.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step
            k=k+1
        else:
            arch7.partial_fit(X_train, y_train,classes=np.unique(class_names))#1 step, no need to pass class argument after 1st iteration
            k=k+1
        if (k%10==0):#at every 10 step, save loss and accuracies
            trainingaccuracyarch7item=arch7.score(X_train, y_train, sample_weight=None)#take training accuracy score 
            trainingaccuracyarch7[t].append(trainingaccuracyarch1item)
            validationaccuracyarch7item=arch7.score(X_validate,y_validate)#take validation accuracy score
            validationaccuracyarch7[t].append(validationaccuracyarch1item)
            traininglossarch7item=arch7.loss_#take training loss 
            traininglossarch7[t].append(traininglossarch1item)
            idx = np.random.permutation(len(X_train))
            X_train,y_train = X_train[idx], y_train[idx]#shuffle at each 10th approach (iteration, step)
    print("done")
    k=0
    weightsofthefirsthiddenlayerarch7item=arch7.coefs_[0]
    weightsofthefirsthiddenlayerarch7.append(weightsofthefirsthiddenlayerarch7item)
    testaccuracyarch7item=arch7.score(test_images,test_labels)
    testaccuracyarch7.append(testaccuracyarch3item)
    arch7 = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16),batch_size=500, verbose=True)#initialise arch7



traininglossarch7avg = []#average curves
trainingaccuracyarch7avg=[]
validationaccuracyarch7avg=[]
for j in range (5400):#calculating average validation, training accuracy and training loss
    for t in range (10):
        traininglossarch7sum=traininglossarch7sum+traininglossarch7[t][j]
        trainingaccuracyarch7sum=trainingaccuracyarch7sum+trainingaccuracyarch7[t][j]
        validationaccuracyarch7sum=validationaccuracyarch7sum+validationaccuracyarch7[t][j]
    traininglossarch7avg.append(traininglossarch7sum/10.0)
    trainingaccuracyarch7avg.append(trainingaccuracyarch7sum/10.0)
    validationaccuracyarch7avg.append(validationaccuracyarch7sum/10.0)

for t in range (10):

    testaccuracyarch7sum=testaccuracyarch7sum+testaccuracyarch7[t]
    weightsofthefirsthiddenlayerarch7sum=weightsofthefirsthiddenlayerarch7sum+weightsofthefirsthiddenlayerarch7[t]
testaccuracyarch7avg=testaccuracyarch7sum/10.0
weightsofthefirsthiddenlayerarch7avg=weightsofthefirsthiddenlayerarch7sum/10.0