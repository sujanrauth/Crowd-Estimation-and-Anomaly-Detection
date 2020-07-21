import os,glob
from flask import Flask,render_template,redirect,request
from werkzeug.utils import secure_filename
import tensorflow as tf
import shutil
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as c
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
from model import CSRNet
import torch
from torchvision import transforms
import cv2
import subprocess
import datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

UPLOAD_FOLDER='./static/uploads'
RESULT_FOLDER='./static/result'
model_path = "./model.h5"
allowed_ext=["jpg","jpeg","png","mp4"]


app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
vidflag=0
th=[0,0,0,0,0]
count_list=[0,0,0,0,0]
color_list=[0,0,0,0,0]
vidcount=list()
vidcolor=list()
violencemodel = load_model(model_path)
model = CSRNet()
checkpoint = torch.load('6_253model_best.pth.tar',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        # response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Cache-Control"] ="no-store"
        # response.headers["Expires"] = 0
        # response.headers["Pragma"] = "no-cache"
        return response

@app.route('/home')
@app.route('/')
def index():
    files=glob.glob(UPLOAD_FOLDER+"/*")
    for f in files:
        os.remove(f)
    resultfiles=glob.glob("./static/result/*")
    for f in resultfiles:
        os.remove(f)
    vidcount.clear()
    vidcolor.clear()

    return render_template('index.html')

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method=="POST":
        q1=request.form['q1']
        q2=request.form['q2']
        q3=request.form['q3']
        q4=request.form['q4']
        oq=request.form['oq']
        with open("{}/threshold.txt".format(UPLOAD_FOLDER),'w') as f:
            f.write(q1+'\n')
            f.write(q2+'\n')
            f.write(q3+'\n')
            f.write(q4+'\n')
            f.write(oq)
        file=request.files['file']
        if validatefile(file.filename)==True:
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            message="Upload Successfull"
            filepath='.'+UPLOAD_FOLDER+'/'+filename
            if 'mp4' in filename:
                filetype='video'
                global vidflag
                vidflag=1
                result_ready=model_run(vidflag)
            else:
                filetype='image'
                result_ready=model_run(vidflag)

            return render_template('upload.html',message=message,filepath=filepath,filetype=filetype)
        else:
            message="file not supported,supoorted files['jpg,jpeg,png,mp4']"

            return render_template('index.html',message=message)


def get_threshold():
    with open('./static/uploads/threshold.txt', 'r') as f:
        t = f.read().splitlines()
    for i in range(0, len(t)):
        th[i] = int(t[i])
    print(th)
    return

def th_check(num,j):
  if num<th[j]:
    return "green"
  elif num>th[j] and num<(th[j]+(th[j]*0.1)):
    return "darkorange"
  else:
    return "darkred"

def get_vidlength(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_vidframes(filename):

    len=get_vidlength(filename)
    vidcap = cv2.VideoCapture(filename)
    fps=int(vidcap.get(cv2.CAP_PROP_FPS)) #getting the frames per sec of video

    total_frames=int(len*fps)
    req_frame=total_frames//2.5
    folder_path=UPLOAD_FOLDER
    vidcap.set(1,req_frame); # Where frame_no is the frame you want
    ret, frame = vidcap.read() # Read the frame
    cv2.imwrite(folder_path+'/frame1.jpeg', frame) # show frame on window
    vidcap.set(1,req_frame*2); # Where frame_no is the frame you want
    ret, frame = vidcap.read() # Read the frame
    cv2.imwrite(folder_path+'/frame2.jpeg', frame) # show frame on window
    # vidcap.set(1,req_frame*3); # Where frame_no is the frame you want
    # ret, frame = vidcap.read() # Read the frame
    # cv2.imwrite(folder_path+'/frame3.jpeg', frame)
    print("done")
    return

def model_run(vidflag):

    get_threshold()
    if vidflag==0:
        f=glob.glob('./static/uploads/*[.png,.jpg,.jpeg]')
        print(f)
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
        photo=Image.open(f[0])
        img = transform(photo.convert('RGB'))
        output = model(img.unsqueeze(0))
        number=int(output.detach().sum().numpy())
        temp = np.asarray(output.detach().reshape(output.detach().shape[2],output.detach().shape[3]))
        del output
        n,m=temp.shape
        n=int(n/4)

        basewidth = m
        wpercent = (basewidth/float(photo.size[0]))
        hsize = int((float(photo.size[1])*float(wpercent)))
        photo = photo.resize((basewidth,hsize), Image.ANTIALIAS)
        photo.save(RESULT_FOLDER+'/13.jpeg')

        st=temp[0:n,:]
        nd=temp[n:2*n,:]
        rd=temp[2*n:3*n,:]
        th=temp[3*n:4*n,:]
        nst=abs(int(np.sum(st)))
        nnd=abs(int(np.sum(nd)))
        nrd=abs(int(np.sum(rd)))
        nth=abs(int(np.sum(th)))

        del photo
        plt.imsave(RESULT_FOLDER+'/2.jpeg',st,cmap=c.jet)
        plt.imsave(RESULT_FOLDER+'/5.jpeg',nd,cmap=c.jet)
        plt.imsave(RESULT_FOLDER+'/8.jpeg',rd,cmap=c.jet)
        plt.imsave(RESULT_FOLDER+'/11.jpeg',th,cmap=c.jet)
        plt.imsave(RESULT_FOLDER+'/14.jpeg',temp,cmap=c.jet)

        del temp

        im=mpimg.imread(RESULT_FOLDER+'/13.jpeg')
        x,y=im.shape[:2]
        xx=int(x/4)

        ost=im[0:xx,:]
        ond=im[xx:2*xx,:]
        ordd=im[2*xx:3*xx,:]
        oth=im[3*xx:4*xx,:]

        plt.imsave(RESULT_FOLDER+'/1.jpeg',ost)
        plt.imsave(RESULT_FOLDER+'/4.jpeg',ond)
        plt.imsave(RESULT_FOLDER+'/7.jpeg',ordd)
        plt.imsave(RESULT_FOLDER+'/10.jpeg',oth)

        del im

        t1=th_check(nst,0)
        color_list[0]=t1
        count_list[0]=nst

        t2=th_check(nnd,1)
        color_list[1]=t2
        count_list[1]=nnd

        t3=th_check(nrd,2)
        color_list[2]=t3
        count_list[2]=nrd

        t4=th_check(nth,3)
        color_list[3]=t4
        count_list[3]=nth

        t5=th_check(number,4)
        color_list[4]=t5
        count_list[4]=number

        print(color_list)
        print(count_list)
        return True

    elif vidflag==1:
        f=glob.glob('./static/uploads/*[.mp4]')
        get_vidframes(f[0])
        # frames=glob.glob('./static/uploads/*[.jpg]')
        for i in range(1,3):
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
            photo=Image.open('./static/uploads/frame'+str(i)+'.jpeg')
            img = transform(photo.convert('RGB'))
            output = model(img.unsqueeze(0))
            number=int(output.detach().sum().numpy())
            temp = np.asarray(output.detach().reshape(output.detach().shape[2],output.detach().shape[3]))
            del output
            del img
            n,m=temp.shape
            n=int(n/4)

            basewidth = m
            wpercent = (basewidth/float(photo.size[0]))
            hsize = int((float(photo.size[1])*float(wpercent)))
            photo = photo.resize((basewidth,hsize), Image.ANTIALIAS)
            photo.save(RESULT_FOLDER+'/'+str(i)+'13.jpeg')
            del photo
            st=temp[0:n,:]
            nd=temp[n:2*n,:]
            rd=temp[2*n:3*n,:]
            th=temp[3*n:4*n,:]
            nst=abs(int(np.sum(st)))
            nnd=abs(int(np.sum(nd)))
            nrd=abs(int(np.sum(rd)))
            nth=abs(int(np.sum(th)))

            plt.imsave(RESULT_FOLDER+'/'+str(i)+'2.jpeg',st,cmap=c.jet)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'5.jpeg',nd,cmap=c.jet)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'8.jpeg',rd,cmap=c.jet)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'11.jpeg',th,cmap=c.jet)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'14.jpeg',temp,cmap=c.jet)
            del temp
            im=mpimg.imread(RESULT_FOLDER+'/'+str(i)+'13.jpeg')
            x,y=im.shape[:2]
            xx=int(x/4)

            ost=im[0:xx,:]
            ond=im[xx:2*xx,:]
            ordd=im[2*xx:3*xx,:]
            oth=im[3*xx:4*xx,:]
            del im
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'1.jpeg',ost)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'4.jpeg',ond)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'7.jpeg',ordd)
            plt.imsave(RESULT_FOLDER+'/'+str(i)+'10.jpeg',oth)

            t1=th_check(nst,0)
            vidcolor.append(t1)
            vidcount.append(nst)

            t2=th_check(nnd,1)
            vidcolor.append(t2)
            vidcount.append(nnd)

            t3=th_check(nrd,2)
            vidcolor.append(t3)
            vidcount.append(nrd)

            t4=th_check(nth,3)
            vidcolor.append(t4)
            vidcount.append(nth)

            t5=th_check(number,4)
            vidcolor.append(t5)
            vidcount.append(number)

        print(vidcolor)
        print(vidcount)
        return True

@app.route('/anomaly',methods=['GET','POST'])
def anomaly():
    if request.method=='POST':
        file=request.files['file']
        if 'mp4' in file.filename.rsplit('.')[1]:
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            message="upload succesfull"
            filepath='.'+UPLOAD_FOLDER+'/'+filename
            filetype='video'


            outcome=anomodel()
            return render_template('home.html')
        else:
            message="file not supported,supoorted file:mp4"

            return render_template('index.html',message=message)

def my_predict(model, image_input, size):
  img = image.load_img(image_input, target_size=(size, size))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  result = violencemodel.predict(img, batch_size=1)
  return np.argmax(result)


def anomodel():
    # video_path = "./testvideo.mp4"
    f=glob.glob('./static/uploads/*[.mp4]')
    model_path = "./model.h5"
    test_path = "./test/"
    out_path = "./output/"

    #Creating the test and output directory
    os.mkdir(test_path)
    os.mkdir(out_path)

    count = 0
    vidcap = cv2.VideoCapture(f[0])
    fps=int(vidcap.get(cv2.CAP_PROP_FPS))
    success,im = vidcap.read()
    while success:
        if (count%fps)==0:
            cv2.imwrite(test_path + "%d.jpg" % count, im)     # save frame as JPEG file
        success,im = vidcap.read()
        count+=1

    print("done")
    # model = load_model(model_path)
    files=[]
    for i in os.listdir(test_path):
      files.append(int(i.split('.')[0]))
    test_sorted = [str(i)+".jpg" for i in sorted(files)]

    SAMPLE_SIZE = 224
    VIEW_SIZE = (600,400)

    for files in test_sorted:
      filename = os.path.join(test_path, files)
      img = cv2.imread(filename)
      img = cv2.resize(img,(VIEW_SIZE[0],VIEW_SIZE[1]),interpolation = cv2.INTER_AREA)
      label = my_predict(model, filename, SAMPLE_SIZE)
      if not label:
        # print("Violence Detected at:",datetime.datetime.now())
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img,"Violence Detected!",(5,25), font, 1,(255,255,255),2)
    #   cv2.imshow("Footage",img)
      cv2.imwrite(os.path.join(out_path, files),img)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()

    img_array = []

    for filename in test_sorted:
        img = cv2.imread(os.path.join(out_path,filename))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(RESULT_FOLDER+'/output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print("done")
    #Deleting created image frames:
    shutil.rmtree(out_path)
    shutil.rmtree(test_path)

    return True

@app.route('/result',methods=['GET','POST'])
def result():
    # return render_template('result.html')

    if vidflag==0:
        return render_template('imgresult2.html' ,colorlist=color_list,countlist=count_list)
    else:
        return render_template('vidresult2.html' ,vidcolor=vidcolor,vidcount=vidcount)

@app.route('/anomaly_result',methods=['GET','POST'])
def anomaly_result():
    return render_template('home.html')

def validatefile(filename):
    if filename.rsplit('.')[1] not in allowed_ext:
        return False
    else:
        return True

if __name__=="__main__":
    app.secret_key = 'super secret key'
    app.run(debug=True)
