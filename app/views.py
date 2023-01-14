from flask import render_template,request
import os 
from app.face_recognisation import face_recognisation
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as matimg

upload_file='static/upload'
def home():
    return render_template('home.html')
    #return "hello flask you are in"
def app():
    return render_template('app.html')
def gender():
    if request.method == 'POST':
        f=request.files['image_name']
        filename=f.filename
        path=os.path.join(upload_file,filename)
        f.save(path)  #save the image into the upload

        #get prediction
        image,pred_score=face_recognisation(path)
        print('predicted successfully')
        pred_file='predicted_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_file}',image)
        #generate report
        report=[]
        for i,obj in enumerate(pred_score):
            img_gr=obj['roi'] #gray image
            eigen_img=obj['eigen_image'] #eigen image
            gender_name=obj['prediction_name'] #name (male or female) 
            score=round(obj['score']*100,2) #score
            # show the image
            img_gr_name=f'roi_{i}.jpg'
            egn_img_name=f'egn_{i}.jpg'
            matimg.imsave(f'./static/predict/{img_gr_name}',img_gr,cmap='gray')
            matimg.imsave(f'./static/predict/{egn_img_name}',eigen_img,cmap='gray')
            #save in report
            report.append([img_gr_name,egn_img_name,gender_name,score])
        return render_template('gender.html',fileupload=True,report=report)

    return render_template('gender.html',fileupload=False)
