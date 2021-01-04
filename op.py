from sql import db,Student

# 增
def add(name, idCard, idImgPath, cetImgPath, faceImgPath):
    s=Student(name=name,idCard=idCard,idImgPath=idImgPath,cetImgPath=cetImgPath,faceImgPath=faceImgPath)
    db.session.add(s)
    db.session.commit()
add('meteor', '111111111111111', 'cscssccsc', 'cscscc', 'scscsc')
# 删
def delete():
    result = Student.query.filter(Student.id == 1).first()
    db.session.delete(result)
    db.session.commit()
# 改
def gai(id):
    stu = Student.query.filter(Student.id == 1).update({"name":"张旭灿"})
    db.session.commit()
    print(stu)

# 查
def get(id):
    stu = Student.query.get(id)
    print(stu)

def get_all():
    stu = Student.query.all()
    print(stu)

def get_filter():
    stu = Student.query.filter(Student.name == "刘星宇").all()#.first()
    print(stu)
# result = Article.query.filter(Article.title=='mark').first()
# print(result.content)
# add()
# delete()
