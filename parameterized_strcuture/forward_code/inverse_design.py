# import tensorflow as tfs
import os
os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input,Dense,concatenate,Lambda,Add,Multiply
from keras.callbacks import ModelCheckpoint
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=None)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

d1 = np.loadtxt("../data/i-dataset/x_et_train_x_S.csv",delimiter=',', dtype='float32')
d2 = np.loadtxt("../data/i-dataset/y_et_train_x_S.csv",delimiter=',', dtype='float32')
d3 = np.loadtxt("../data/i-dataset/x_et_test_x_S.csv",delimiter=',', dtype='float32')
d4 = np.loadtxt("../data/i-dataset/y_et_test_x_S.csv",delimiter=',', dtype='float32')
x1 = d1[:,0:4]
x2 = d1[:,4:5]
x3 = d3[:,0:4]
x4 = d3[:,4:5]
y=d2[:,1:2]
y2=d4[:,1:2]
# maxx1=np.max(x1,axis=0)
# minx1=np.min(x1,axis=0)
maxx1=np.array([9.5,1.3,2.5,1.2])
minx1=np.array([8,0.3,0.5,0.2])
maxx2=14
minx2=2
x1=(x1-minx1)/(maxx1-minx1)*2-1
x2_1=(x2-0.1-minx2)/(maxx2-minx2)*2-1
x2_2=(x2-minx2)/(maxx2-minx2)*2-1
x2_3=(x2+0.1-minx2)/(maxx2-minx2)*2-1
x3=(x3-minx1)/(maxx1-minx1)*2-1
x4_1=(x4-0.1-minx2)/(maxx2-minx2)*2-1
x4_2=(x4-minx2)/(maxx2-minx2)*2-1
x4_3=(x4+0.1-minx2)/(maxx2-minx2)*2-1
y=y*2-1
y2=y2*2-1
X_train_11 = (x1, x2_1)
X_train_12 = (x1, x2_2)
X_train_13 = (x1, x2_3)
y_2_train = y
X_test_11 = (x3, x4_1)
X_test_12 = (x3, x4_2)
X_test_13 = (x3, x4_3)
y_2_test = y2
data1 = dde.data.Triple(X_train=X_train_12, y_train=y_2_train, X_test=X_test_12, y_test=y_2_test)
# electro-model
### this part is from the demo of DeepXDE
m1 = 4
dim_x1 = 1
net1 = dde.nn.DeepONet(
    [m1, 300, 300, 300, 300],
    [dim_x1, 300, 300, 300, 300],
    "relu",
    "Glorot normal",
)
# Define a Model
model1 = dde.Model(data1, net1)
# Compile and Train
model1.compile("adam", lr=0.001, loss='MSE')
##############################################
model1.restore("..\\data\\em-model\\m_300_4-40000.ckpt")
net1.trainable=False
Inputx=Input(shape=(4,))
Inputf2=Input(shape=(1,))
Inputf1=Lambda(lambda x: x-0.1/(maxx2-minx2)*2)(Inputf2)
Inputf3=Lambda(lambda x: x+0.1/(maxx2-minx2)*2)(Inputf2)
lab1=net1.call((Inputx,Inputf1),training=False)
lab2=net1.call((Inputx,Inputf2),training=False)
lab3=net1.call((Inputx,Inputf3),training=False)
# electro-thermal model data
y_1_train1=model1.predict(X_train_11)
y_1_train2=model1.predict(X_train_12)
y_1_train3=model1.predict(X_train_13)
y_1_test1=model1.predict(X_test_11)
y_1_test2=model1.predict(X_test_12)
y_1_test3=model1.predict(X_test_13)
y_train=y_2_train-y_1_train2
y_test=y_2_test-y_1_test2
x_2_train=np.c_[x2_2,y_1_train1]
x_2_train=np.c_[x_2_train,y_1_train2]
x_2_train=np.c_[x_2_train,y_1_train3]
x_2_test=np.c_[x4_2,y_1_test1]
x_2_test=np.c_[x_2_test,y_1_test2]
x_2_test=np.c_[x_2_test,y_1_test3]
X_train = (x1, x_2_train)
X_test = (x3, x_2_test)
data2 = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# electro-thermal model
### this part is from the demo of DeepXDE
m2 = 4
dim_x2 = 4
net2 = dde.nn.DeepONet(
    [m2, 300, 300, 300],
    [dim_x2, 300, 300],
    "relu",
    "Glorot normal",
    # regularization = ["l2", 0.01]
)
net2.trainable=False
model2 = dde.Model(data2, net2)
##############################################
model2.restore("..\\data\\e-t-model\\m_re_1-1000.ckpt")
combine = concatenate([Inputf2,lab1,lab2,lab3])
Out1=net2.call((Inputx,combine),training=False)
Out1=Add()([Out1,lab2])
model_v = Model(inputs=[Inputx,Inputf2], outputs=Out1)
# model_v.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
model_v.trainable=False
# inverse model
inS=Input(shape=(61,))
a1=Dense(120, kernel_initializer='he_normal', activation='elu')(inS)
a1=Dense(120, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(60, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(60, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu')(a1)
b1 = Dense(1,kernel_initializer='uniform', activation='tanh')(a1)
b2 = Dense(1,kernel_initializer='uniform', activation='tanh')(a1)
b3 = Dense(1,kernel_initializer='uniform', activation='tanh')(a1)
b4 = Dense(1,kernel_initializer='uniform', activation='tanh')(a1)
b = concatenate([b1,b2,b3,b4])
z1=Model(inputs=inS,outputs=b)
z1.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
# physic val
c1=Lambda(lambda x: 0.0721022-0.34994227*x)(b1)
c2=Lambda(lambda x: 0.30063257*x)(b2)
c3=Lambda(lambda x: 0.18642503*x)(b3)
c4=Lambda(lambda x: 0.19676195*x)(b4)
c5=Lambda(lambda x: -0.05123273*x**2)(b1)
c6=Multiply()([b1, b2])
c6=Lambda(lambda x: -0.01159249*x)(c6)
c7=Multiply()([b1, b3])
c7=Lambda(lambda x: 0.03361623*x)(c7)
c8=Multiply()([b1, b4])
c8=Lambda(lambda x: -0.06290007*x)(c8)
c9=Lambda(lambda x: -0.02030303*x**2)(b2)
c10=Multiply()([b2, b3])
c10=Lambda(lambda x: 0.02372525*x)(c10)
c11=Multiply()([b2, b4])
c11=Lambda(lambda x: 0.00998978*x)(c11)
c12=Lambda(lambda x: 0.0118433*x**2)(b3)
c13=Multiply()([b3, b4])
c13=Lambda(lambda x: 0.06386705*x)(c13)
c14=Lambda(lambda x: -0.02462226*x**2)(b4)
c=Add()([c1,c2,c3,c4])
# forward val
fstep=0.6
in1=Input(shape=(1,))
in2=Lambda(lambda x: x+1*fstep/(maxx2-minx2)*2)(in1)
in3=Lambda(lambda x: x+2*fstep/(maxx2-minx2)*2)(in1)
in4=Lambda(lambda x: x+3*fstep/(maxx2-minx2)*2)(in1)
in5=Lambda(lambda x: x+4*fstep/(maxx2-minx2)*2)(in1)
in6=Lambda(lambda x: x+5*fstep/(maxx2-minx2)*2)(in1)
in7=Lambda(lambda x: x+6*fstep/(maxx2-minx2)*2)(in1)
in8=Lambda(lambda x: x+7*fstep/(maxx2-minx2)*2)(in1)
in9=Lambda(lambda x: x+8*fstep/(maxx2-minx2)*2)(in1)
in10=Lambda(lambda x: x+9*fstep/(maxx2-minx2)*2)(in1)
in11=Lambda(lambda x: x+10*fstep/(maxx2-minx2)*2)(in1)
in12=Lambda(lambda x: x+11*fstep/(maxx2-minx2)*2)(in1)
in13=Lambda(lambda x: x+12*fstep/(maxx2-minx2)*2)(in1)
in14=Lambda(lambda x: x+13*fstep/(maxx2-minx2)*2)(in1)
in15=Lambda(lambda x: x+14*fstep/(maxx2-minx2)*2)(in1)
in16=Lambda(lambda x: x+15*fstep/(maxx2-minx2)*2)(in1)
in17=Lambda(lambda x: x+16*fstep/(maxx2-minx2)*2)(in1)
in18=Lambda(lambda x: x+17*fstep/(maxx2-minx2)*2)(in1)
in19=Lambda(lambda x: x+18*fstep/(maxx2-minx2)*2)(in1)
outv1=model_v([z1.output,in1])
outv2=model_v([z1.output,in2])
outv3=model_v([z1.output,in3])
outv4=model_v([z1.output,in4])
outv5=model_v([z1.output,in5])
outv6=model_v([z1.output,in6])
outv7=model_v([z1.output,in7])
outv8=model_v([z1.output,in8])
outv9=model_v([z1.output,in9])
outv10=model_v([z1.output,in10])
outv11=model_v([z1.output,in11])
outv12=model_v([z1.output,in12])
outv13=model_v([z1.output,in13])
outv14=model_v([z1.output,in14])
outv15=model_v([z1.output,in15])
outv16=model_v([z1.output,in16])
outv17=model_v([z1.output,in17])
outv18=model_v([z1.output,in18])
outv19=model_v([z1.output,in19])
out_v = concatenate([outv1,outv2,outv3,outv4,outv5,outv6,outv7,outv8,outv9,outv10,
                     outv11,outv12,outv13,outv14,outv15,outv16,outv17,outv18,outv19])
m_combine=Model(inputs=[inS,in1], outputs=[b,c,out_v])
m_combine.compile(loss=['mse','mse','mse'],
                  loss_weights=[0.8,0.1,0.1],
                  optimizer='adam',
                  metrics=['mse'])
train=False
filepath_1='../data/i-model/model_combine_weight.h5'
filepath_2='../data/i-model/model_inverse_weight.h5'
filepath_3='../data/i-model/model_combine_finalstep_weight.h5'
model_name='../data/i-model/inverse_model.keras'
callbacks = [
    ModelCheckpoint(filepath_1, monitor='val_loss', save_weights_only=True,verbose=1,save_best_only=True,period=1),
]
dataset1=np.loadtxt("../data/i-dataset/inverse_design.csv",delimiter=',', dtype='float32')
dataset2=np.loadtxt("../data/i-dataset/deeponet_val.csv",delimiter=',', dtype='float32')
dataset3=np.loadtxt("../data/i-dataset/reg.csv",delimiter=',', dtype='float32')
stru=(dataset1[:,0:4]-minx1)/(maxx1-minx1)*2-1
s21=dataset1[:,4:65]*2-1
f_sample=(dataset2[:,0:19]-minx2)/(maxx2-minx2)*2-1
S_sample=dataset2[:,19:38]*2-1
fy=dataset3[:,4].reshape(len(dataset3[:,4]),1)
train_num=635
# train
stru_1=stru[0:train_num,:]
S21_1=s21[0:train_num,:]
f_1=f_sample[0:train_num,:]
S_1=S_sample[0:train_num,:]
fy1=fy[0:train_num,:]
# val
stru_2=stru[train_num:710,:]
S21_2=s21[train_num:710,:]
f_2=f_sample[train_num:710,:]
S_2=S_sample[train_num:710,:]
fy2=fy[train_num:710,:]
if train:
    history=m_combine.fit(x=[S21_1,f_1[:,0]],
                          y=[stru_1,fy1,S_1], epochs=30, batch_size=100, shuffle=True, verbose=1,
                          validation_data=([S21_2,f_2[:,0]], [stru_2,fy2,S_2]),
                          callbacks=callbacks)
    loss_train=np.array(history.history['loss'])
    loss_test=np.array(history.history['val_loss'])
    m_combine.save_weights(filepath_3)
    m_combine.load_weights(filepath_1)
    z1.save_weights(filepath_2)    
    z1.save(model_name)    
    m_combine.load_weights(filepath_3)
    testset=np.loadtxt("../data/i-dataset/inverse_test_2.csv",delimiter=',', dtype='float32')    
    test=testset[:,1].reshape(1,len(testset[:,1]))*2-1
    testy=(z1.predict(test)+1)/2*(maxx1-minx1)+minx1
    # model.save("G:\electro-thermal\electro\model_total", protocol="backend")
else:    
    m_combine.load_weights(filepath_1)
    [yy,ff,ss]= m_combine.predict([S21_2,f_2[:,0]])
    y_true=stru_2
    s_true=S_2
    ff_true=fy2
    e1=np.linalg.norm(yy-y_true,ord=2)/np.linalg.norm(y_true,ord=2)
    e2=np.linalg.norm(ff-ff_true,ord=2)/np.linalg.norm(ff_true,ord=2)
    e3=np.linalg.norm(ss-s_true,ord=2)/np.linalg.norm(s_true,ord=2)    
    testset=np.loadtxt("../data/i-dataset/inverse_test_2.csv",delimiter=',', dtype='float32')    
    test=testset[:,1].reshape(1,len(testset[:,1]))*2-1    
    testy=(z1.predict(test)+1)/2*(maxx1-minx1)+minx1
    yy=(yy+1)/2*(maxx1-minx1)+minx1
    with open("../results/inverse_design_result.txt","w") as f1:
        f1.write(str(testy))
    with open("../results/inverse_test_result.txt","w") as f2:
        f2.write(str(yy))   
    
