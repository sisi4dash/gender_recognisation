import pandas as pd
import numpy as np
import pickle
import sklearn
import cv2

#loading all the models 
haar=cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_svm=pickle.load(open('./model/ML_model_svm.pickle','rb'))
pca_models=pickle.load(open('./model/pca_dict_model.pickle','rb'))
#pca model contain pca and mean face both so we need to separate both
model_pca=pca_models['pca']
model_meanface=pca_models['mean_face']

#full function
def face_recognisation(file_name,path=True):
    if path:
        img=cv2.imread(file_name)
    else:
        img=file_name
    img_gr=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    faces=haar.detectMultiScale(img_gr,1.5,3)
    predictions=[]
    for x,y,w,h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi=img_gr[y:y+h,x:x+w]
        roi=roi/255.0
        if roi.shape[1]>100:
            roi_resize=cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize=cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        roi_flat=roi_resize.reshape(1,10000)
        roi_mean=roi_flat-model_meanface
        egn_img=model_pca.transform(roi_mean)
        egn_img1=model_pca.inverse_transform(egn_img)
        result=model_svm.predict(egn_img)
        prb_score=model_svm.predict_proba(egn_img)
        #print(result,prb_score)
        prb_score_max=prb_score.max()
        text="%s : %d"%(result[0],prb_score_max*100)
        #print(text)
        #define differnt color for male and female
        if result[0]=='male':
            color=(255,0,255)
        else:
            color=(0,255,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,4)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
        output ={'roi':roi,'eigen_image':egn_img1,'prediction_name':result[0],'score':prb_score_max}
        predictions.append(output)
    return img,predictions
    