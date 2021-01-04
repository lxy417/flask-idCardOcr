from flask_sqlalchemy import SQLAlchemy
import pymysql
from  flask import Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:chenchen417@101.200.221.167:3306/test"
# 动态追踪修改设置，如未设置只会提示警告
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# app.config['SECRET_KEY']='123456'
#查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = False

db =SQLAlchemy(app)

class Student(db.Model):
    __tablename__="student"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64),nullable=False)
    idCard = db.Column(db.String(18),nullable=False)
    idImgPath = db.Column(db.String(64))
    cetImgPath = db.Column(db.String(64))
    faceImgPath = db.Column(db.String(64))
    isReport = db.Column(db.Boolean, doc='是否报道', default=False)

if __name__ == '__main__':
    db.drop_all()
    db.create_all()
