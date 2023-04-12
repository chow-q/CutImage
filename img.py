from flask import Flask, render_template, request
import os
import uuid
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 判断是否有文件上传
        if "file_input" not in request.files:
            return render_template("index.html", error="请选择一个图片上传！")

        file = request.files["file_input"]
        print("##################",request.form)

        # 判断上传的文件类型是否合法
        allowed_extensions = {"jpg", "jpeg", "png", "gif"}
        _, file_extension = os.path.splitext(file.filename)
        if not file_extension[1:] in allowed_extensions:
            return render_template("index.html", error="只允许上传 .jpg、.jpeg、.png、.gif 格式的图片！")

        # 生成一个唯一文件名，避免重复
        image_filename = str(uuid.uuid4()) + file_extension

        # 保存上传的图片
        file.save(os.path.join("static", image_filename))
        
        if 'image' in request.form:
          #人物抠图
          portrait_matting = pipeline(Tasks.portrait_matting,model='damo/cv_unet_image-matting')
          result = portrait_matting(f"static/{image_filename}")
          result_filename = str(uuid.uuid4()) + ".png"
          cv2.imwrite(f"static/{result_filename}", result[OutputKeys.OUTPUT_IMG])
        else:
          #通用抠图
          universal_matting = pipeline(Tasks.universal_matting,model='damo/cv_unet_universal-matting')
          result = universal_matting(f"static/{image_filename}")
          result_filename = str(uuid.uuid4()) + ".png"
          cv2.imwrite(f"static/{result_filename}", result[OutputKeys.OUTPUT_IMG])

        # 显示图片
        image_path = f"static/{image_filename}"
        result_path = f"static/{result_filename}"
        return render_template("index.html", image_path=image_path, result_path=result_path)

    return render_template("index.html")
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
