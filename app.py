from flask import Flask, render_template,request
from flask_cors import CORS
import base64
import os
from idcardocr import process,recognition
# from op import *
from sql import db,Student
import json
app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route('/')
def index():
    return "hello"

@app.route('/uploadidcard', methods=["POST"])
def upload_idCard():
    upload_file = request.files['file']
    file_name = upload_file.filename
    # 文件保存目录
    file_path=r'./static/image/'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        # 随便打开一张其他图片作为结果返回，
        # with open(file_paths, 'rb') as f:
        #     res = base64.b64encode(f.read())
    res = recognition.main(file_name)
    print(res)
    # name = request.form.get('name')
    # age = request.form.get('age')
    # return "user_name = %s, user_age = %s" % (name,age)
    return {"status":1,"image_path":"static/image/"+file_name,"data":res}


@app.route('/upload', methods=["POST"])
def upload_request():
    upload_file = request.files['file']
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path=r'./static/image/'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        # 随便打开一张其他图片作为结果返回，
        with open(file_paths, 'rb') as f:
            res = base64.b64encode(f.read())

    # name = request.form.get('name')
    # age = request.form.get('age')
    # return "user_name = %s, user_age = %s" % (name,age)
    return {"status":1,"image_path":"static/image/"+file_name}

@app.route('/submit', methods=["POST"])
def submit_request():
    name = request.form.get('name')
    idCard = request.form.get('idCard')
    idImgPath = request.form.get('idImgPath')
    cetImgPath = request.form.get('cetImgPath')
    faceImgPath = request.form.get('faceImgPath')
    data = [name,idCard,idImgPath,cetImgPath,faceImgPath]
    # data = [name=request.form.get]
    s = Student(name=name, idCard=idCard, idImgPath=idImgPath, cetImgPath=cetImgPath, faceImgPath=faceImgPath)
    db.session.add(s)
    db.session.commit()
    # add('meteor', '111111111111111', "'cscssccsc'", 'cscscc', 'scscsc')
    # add(name,idCard,idImgPath,cetImgPath,faceImgPath)
    print(data)
    return {"status":1}


if __name__ == '__main__':
    app.run()