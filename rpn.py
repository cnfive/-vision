# -*- coding:utf-8 -*-
#训练图像区域建议方案
#先训练是否是建议的铆点
import tensorflow as tf
import os
from skimage import io,transform
import  xml.dom.minidom
import numpy as np
import math
#import matplotlib.pyplot as plt;
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
from skimage import io,data
import os
from PIL import Image
#读取训练集的图片位置信息
#遍历注释文档
import xml.etree.cElementTree as et


import sys,os


#将所有的图片resize成200*200
w=200
h=200
c=3
#-----------------构建网络----------------------
#占位符
xin=tf.placeholder(tf.float32,shape=[None,w,h,c])
y_out=tf.placeholder(tf.int32,shape=[12,12,18])

#第一个卷积层（100——>50)
conv1=tf.layers.conv2d(
      inputs=xin,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu ,
     
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#第二个卷积层(50->25)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#第三个卷积层(25->12)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#第四个卷积层(12->6)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

print("pool4.shape",pool4)

conv5=tf.layers.conv2d(
      inputs=pool4,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
print("conv5.shape:",conv5)

conv_rpn=tf.layers.conv2d(
      inputs=pool4,
      filters=18,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

with tf.variable_scope('outputs'):
      rpn=tf.arg_max(conv_rpn,dimension=3 )


      imgg=tf.image.convert_image_dtype(rpn, dtype=tf.float32)
print("conv_rpn.shape:",conv_rpn)

#rpn_reshape=tf.reshape(conv_rpn,[2,12])



#deconv1 = tf.nn.conv2d_transpose(conv6, wt, [FLAGS.batch_size, 130, 100, 1], [1, 10, 10, 1], 'SAME')  
#re1 = tf.reshape(pool4, [-1, 12 * 12* 128])

#---------------------------网络结束---------------------------


rootdir='/media/hadoop/文档/voc2012/Annotations/'

def IOU(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。·
    """
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2]-Reframe[0];
    height1 = Reframe[3]-Reframe[1];

    x2 = GTframe[0];
    y2 = GTframe[1];
    width2 = GTframe[2]-GTframe[0];
    height2 = GTframe[3]-GTframe[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0  :
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height; # 两矩形相交面积
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    # return IOU
    return ratio,Reframe,GTframe



count=0
imgs=[]

w2=12
h2=12
tag_y=[]
list = os.listdir(rootdir) 



box_true=[]
label_list=[]

for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    print(path)

    tree=et.parse(path)
    root=tree.getroot()

    filename=root.find('filename').text
    print (filename)
    pic_size=root.find('size')
    pic_width=pic_size.find("width").text
    print("width:",pic_width)
    pic_height=pic_size.find("height").text
    print("pic_height:",pic_height)
    filepath="/media/hadoop/文档/voc2012/jpegimages/"+filename
    img=io.imread(filepath)
    img=transform.resize(img,(w,h))
    imgs.append(img)
    tag=np.zeros((12,12))
    box_true_2=[]

    #计算预想的9个不同尺寸的框  128*128,128*256,256*128 ，256*256,256*512,512*256,512*512,1024*512,512*1024
    #计算缩放后的对应9个铆点的实际尺寸和位置

   # (w/int(pic_width))   (h/int(pic_height))
    #for g in range(12):
    w1=128*(w/int(pic_width))
    h1=128*(h/int(pic_height))
    w2=256*(w/int(pic_width))
    h2=256*(h/int(pic_height))
    w3=512*(w/int(pic_width))
    h3=512*(h/int(pic_height))
    w4=1024*(w/int(pic_width))
    h4=1024*(h/int(pic_height))
    
    b1=[w1*(w/int(pic_width)),h1*(h/int(pic_height))] # 128*128
    b2=[w2*(w/int(pic_width)),h1*(h/int(pic_height))]  # 256*128
    b3=[w1*(w/int(pic_width)),h2*(h/int(pic_height))]  # 128*256
    b4=[w2*(w/int(pic_width)),h2*(h/int(pic_height))]  # 256*256
    b5=[w2*(w/int(pic_width)),h3*(h/int(pic_height))]       # 256*512
    b6=[w3*(w/int(pic_width)),h2*(h/int(pic_height))]        #512*256
    b7=[w3*(w/int(pic_width)),h3*(h/int(pic_height))]  # 512*512
    b8=[w3*(w/int(pic_width)),h4*(h/int(pic_height))]  #512*1024
    b9=[w4*(w/int(pic_width)),h3*(h/int(pic_height))]  #1024*512
    print("b1:",b1)
    print("b2:",b2)
    print("b3:",b3)
    print("b4:",b4)
    print("b5:",b5)
    print("b6:",b6)
    print("b7:",b7)
    print("b8:",b8)
    print("b9:",b9)
    #计算每组盒子的对应坐标
    #w1/2  # r=w/24 计算点中心距离
    rx=w/24
    ry=h/24
    d1x1=rx-w1/2
    d1y1=ry-h1/2
    print("d1x1:",d1x1)
    print("d1y1:",d1y1)
    d1x2=d1x1+w1*(w/int(pic_width))
    d1y2=d1y1+h1*(h/int(pic_height))
    #计算box 坐标
    box=[None]*9
    boxid=0
    box[boxid]=[d1x1,d1y1,d1x2,d1y2]
    print(box[0])

    boxid=boxid+1
    #box2
    d2x1=rx-w2/2
    d2y1=ry-h1/2
    d2x2=d2x1+w2*(w/int(pic_width))
    d2y2=d2y1+h1*(h/int(pic_height))
    box[boxid]=[d2x1,d2y1,d2x2,d2y2]
    print(box[1])
    boxid=boxid+1
    d3x1=rx-w1/2
    d3y1=ry-h2/2
    d3x2=d3x1+w1*(w/int(pic_width))
    d3y2=d3y1+h2*(h/int(pic_height))
    box[boxid]=[d3x1,d3y1,d3x2,d3y2]
    print(box[2])
    boxid=boxid+1
    d4x1=rx-w2/2
    d4y1=ry-h2/2
    d4x2=d4x1+w2*(w/int(pic_width))
    d4y2=d4y1+h2*(h/int(pic_height))
    box[boxid]=[d4x1,d4y1,d4x2,d4y2]
    print(box[3])
    boxid=boxid+1
    d5x1=rx-w2/2
    d5y1=ry-h3/2
    d5x2=d5x1+w2*(w/int(pic_width))
    d5y2=d5y1+h3*(h/int(pic_height))
    box[boxid]=[d5x1,d5y1,d5x2,d5y2]
    print(box[4])
    boxid=boxid+1
    #w3 ,h2
    d6x1=rx-w3/2
    d6y1=ry-h2/2
    d6x2=d6x1+w3*(w/int(pic_width))
    d6y2=d6y1+h2*(h/int(pic_height))
    box[boxid]=[d6x1,d6y1,d6x2,d6y2]
    print(box[5])
    boxid=boxid+1
    #w3 ,h3
    d7x1=rx-w3/2
    d7y1=ry-h3/2
    d7x2=d7x1+w3*(w/int(pic_width))
    d7y2=d7y1+h3*(h/int(pic_height))
    box[boxid]=[d7x1,d7y1,d7x2,d7y2]
    print(box[6])
    boxid=boxid+1
   #w3,h4
    d8x1=rx-w3/2
    d8y1=ry-h4/2
    d8x2=d8x1+w3*(w/int(pic_width))
    d8y2=d8y1+h4*(h/int(pic_height))
    box[boxid]=[d8x1,d8y1,d8x2,d8y2]
    print(box[7])
    boxid=boxid+1
   #w4,h3
    d9x1=rx-w4/2
    d9y1=ry-h3/2
    d9x2=d9x1+w4*(w/int(pic_width))
    d9y2=d9y1+h3*(h/int(pic_height))
    box[boxid]=[d9x1,d9y1,d9x2,d9y2]
    print(box[8])
    box_n=[]

    for j in range(0,9):
         b_o=box[j]
         w1=b_o[2]-b_o[0]
 
         h1=b_o[3]-b_o[1]
         box_n.append([math.ceil(w1),math.ceil(h1)])
    print(box_n)

  # 构建9×12×12 的坐标位置
    box_group=[]
    box_group2=[]
    for n in range(12):
        for j in range(0,9):
             box_u=box[j]
             box_z=[None]*4
             box_z[0]=box_u[0]+(w/12)*(n+1)
             box_z[1]=box_u[1]
             box_z[2]=box_u[2]+(w/12)*(n+1)
             box_z[3]=box_u[3]
             box_group.append(box_z)
    for j in box_group:
             box_group2.append(j)
    for n in range(11):
         for j in box_group:
             box_u=j
             #print(box_u)
             box_z=[None]*4
             box_z[0]=box_u[0]
             box_z[1]=box_u[1]+(h/12)*(n+1)
             box_z[2]=box_u[2]
             box_z[3]=box_u[3]+(h/12)*(n+1)
             box_group2.append(box_z)
              
    print(len(box_group2))




  
    for Object in root.findall('object'):
        name=Object.find('name').text
        print (name)
        bndbox=Object.find('bndbox')
        xmin=bndbox.find('xmin').text
        ymin=bndbox.find('ymin').text
        xmax=bndbox.find('xmax').text
        ymax=bndbox.find('ymax').text
        print( xmin,ymin,xmax,ymax)
    #ratio=w/int(pic_width)

    #
    #计算缩放后的标注框位置
        xminn=int(xmin)*(w/int(pic_width))
        print("xmin:",xmin)
        print("xminn:",xminn)
        xmaxx=int(xmax)*(w/int(pic_width))
        print("xmax:",xmax)
        print("xmaxx:",xmaxx)
        yminn=int(ymin)*(h/int(pic_height))
        print("ymin:",ymin)
        print("yminn:",yminn)
        ymaxx=int(ymax)*(h/int(pic_height))
        print("ymax:",ymax)
        print("yamxx:",ymaxx)

        box_true_2.append([xminn,yminn,xmaxx,ymaxx])
     # 计算缩小后的标注框在 12*12 里面的位置
        xminn2=math.ceil(int(xmin)*(w2/int(pic_width)))
        print("xmin:",xmin)
        print("xminn2:",xminn2)
        xmaxx2=math.ceil(int(xmax)*(w2/int(pic_width)))
        print("xmax:",xmax)
        if xmaxx2>12:
           ymaxx2=12
        print("xmaxx2:",xmaxx2)

        yminn2=math.ceil(int(ymin)*(h2/int(pic_height)))
        print("ymin:",ymin)
        print("yminn2:",yminn2)
        ymaxx2=math.ceil(int(ymax)*(h2/int(pic_height)))
        if ymaxx2>12:
           ymaxx2=12
        print("ymax:",ymax)
        print("yamxx2:",ymaxx2) 
        
    #计算每个铆点图片的iou
    iou_list=[]
    iou_list_2=[]
    top_k=50
    iou_dic={}
    iou_dic_2={}
    num=0
    num2=0
    for m in    box_group2:
        for t in box_true_2:
            for v in m:
                if v<0:
                    continue
                if v>h:
                    continue
               
            iou,g_p,ture_p =IOU(m,t)
            if iou>0.12:
                #print("iou:",iou)
                if iou not in iou_list:
                    iou_list.append(iou)
                iou_dic[num]=[iou,g_p,ture_p]
                num=num+1

            if iou<0.11:
                if iou not in iou_list_2:
                    iou_list_2.append(iou)
                
                iou_dic_2[num2]=[iou,g_p,ture_p]
                num2=num2+1



    iou_list_sort=sorted(iou_list,reverse=True)
    #print(iou_list_sort)
    iou_top=[]
    for k in range(top_k):
        list_a=iou_list_sort[k:]
        t=max(list_a)
        print(t)
        iou_top.append(t)
     
              
    #计算iou
    imgsrc=cv2.imread(filepath)
#print(imgsrc)
    plt.figure(8)
    plt.imshow(img)
    currentAxis=plt.gca()
    
    for key in iou_dic:

       for box_i in iou_top:  
 
           if box_i in iou_dic[key]:
              #box=iou_dic[key]

              box=iou_dic[key]
              g_p=box[1]
        #t_p=box[1]
              box_width=g_p[2]-g_p[0]
              box_height=g_p[3]-g_p[1]
              rect=patches.Rectangle((g_p[0], g_p[1]),box_width,box_height,linewidth=1,edgecolor='r',facecolor='none')
              currentAxis.add_patch(rect)  
              t_p=box[2]
              box_width=t_p[2]-t_p[0]
              box_height=t_p[3]-t_p[1]
              rect=patches.Rectangle((g_p[0], g_p[1]),box_width,box_height,linewidth=1,edgecolor='b',facecolor='none')
              currentAxis.add_patch(rect)   
        #输出iou
              t = box_i 
    
              plt.text(g_p[0], g_p[1], t, ha='left', color = "r",rotation=0, wrap=True,bbox = dict(facecolor = "r", alpha = 0.2)) 

    #plt.show()    
   # box_n=[]
 

#生成前景标签，剩下的就是背景标签 ,还原box 在9*12*12的具体位置
    x_y_z=[]
    for key in iou_dic:
     for box_j in iou_top:
       if box_j in iou_dic[key]:
         box=iou_dic[key]
         #box=iou_dic[box_i]
         g_p=box[1]
         print(g_p)
         #计算坐标，计算在那一层

         x1=g_p[0]
         x2=g_p[2]
         y1=g_p[1]
         y2=g_p[3]
         width=math.ceil(x2-x1)
         height=math.ceil(y2-y1)
         xx=x1+(x2-x1)/2
         yy=y1+(y2-y1)/2
         print("xx:",xx)
         print("yy:",yy)
         location_x=math.ceil(xx/(w/12))-1
         location_y=math.ceil(yy/(h/12))-1
         print("x,y:",[location_x,location_y])
         print("width:",width)
         print("height:",height)
         depth=0
         z=0
         for b in box_n:
             
             if b==[width,height]:
                  print("in 9 list at:",depth)
                  z=depth
             depth=depth+1
         x_y_z.append([location_x,location_y,z])
         
    print(x_y_z)  

    print("------------------------------end-a--------------------------------------") 

    bx_by_bz=[]
    for key in iou_dic_2:
     #for box_j in iou_top:
      # if box_j in iou_dic[key]:
         box=iou_dic_2[key]
         #box=iou_dic[box_i]
         g_p=box[1]
         print(g_p)
         #计算坐标，计算在那一层

         x1=g_p[0]
         x2=g_p[2]
         y1=g_p[1]
         y2=g_p[3]
         width=math.ceil(x2-x1)
         height=math.ceil(y2-y1)
         xx=x1+(x2-x1)/2
         yy=y1+(y2-y1)/2
         print("xx:",xx)
         print("yy:",yy)
         location_x=math.ceil(xx/(w/12))-1
         location_y=math.ceil(yy/(h/12))-1
         print("x,y:",[location_x,location_y])
         print("width:",width)
         print("height:",height)
         depth=0
         z=0
         for b in box_n:
             
             if b==[width,height]:
                  print("in 9 list at:",depth)
                  z=depth
             depth=depth+1
         if location_x>-1 and location_y>-1 and z>-1:
             bx_by_bz.append([location_x,location_y,z])
    print("------------------------------end-b--------------------------------------") 
    print(bx_by_bz)  


   
             #print("x:",x)
    labels=np.zeros((12,12,9*2))
    #標注前景
    for num in x_y_z:
        print("num:",num)
        x,y,z=num
        labels[x,y,z*2]=1
        #labels[x,y,z*2]=1
    #標注背景
    for num  in bx_by_bz:
        print("num:",num)
        x,y,z=num
        labels[x,y,z*2-1]=1
        #labels[x,y,z*2]=1
        
    for l in labels:
        print(l)

    print(labels.shape)
    label_list.append(labels)
    
    box_true.append(box_true_2)
    print(tag)
    tag_y.append(tag)


    count=count+1
    print("count:",count)
    if count>30:
        break
    #转换为变形后的标注框大小
print("hello")
#print(box_true)
#  计算预想的9个不同尺寸的框  128*128,128*256,256*128 ，256*256,256*512,512*256,512*512,1024*512,512*1024
#imgsrc=io.imread(filepath)




yo=tf.reshape(y_out,[-1])
rpn_reshape=tf.reshape(conv_rpn,[-1])
#prob=tf.nn.softmax(rpn_reshape, name="prob")
#loss=tf.nn.softmax_cross_entropy_with_logits(logits=prob,labels=yo)
y_out2=tf.image.convert_image_dtype(y_out, dtype=tf.float32)
print("conv_rpn.shape:",conv_rpn.shape)
print("y_out.shape:",y_out.shape)
#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_reshape, labels=yo)

rand = np.random.randint(0,18)
print("rand:",rand)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=conv_rpn,labels=y_out))
saver = tf.train.Saver(max_to_keep=3)
                                                                
#loss=0.001
train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
#train_op= tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
#tf.image.convert_image_dtype([data3[:,:,0],data3[:,:,1],data3[:,:,2]], dtype=tf.uint8)
correct_prediction = tf.equal(conv_rpn,y_out2 )    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.device('/cpu:0'):
     sess=tf.Session()  
     sess=tf.InteractiveSession()
     sess.run(tf.global_variables_initializer())
     for m in range(1000):
         for c in range(30):
             yout=label_list[c]
             img_in=sess.run(tf.expand_dims(imgs[c],0))
             print(yout.shape)
             loss2,_,acc2=sess.run([loss,train_op,acc], feed_dict={xin:img_in ,y_out:yout})
             print(acc2)
             print("loss2=",loss2)
        
         if  loss2<0.001 :
             saver.save(sess, "Model/model.ckpt",global_step=m)













