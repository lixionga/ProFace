from flask import Flask, render_template, request
from exts import db  # 扩展插件

# 导入蓝图
from blueprints.text import bp as text_bp
from blueprints.image import bp as img_bp
from blueprints.video import bp as video_bp
from blueprints.audio import bp as audio_bp
from blueprints.forgery import bp as forgery_bp
from config import config 


app = Flask(__name__)
app.config.from_object(config)  # 绑定配置文件

# 注册蓝图
app.register_blueprint(text_bp)
app.register_blueprint(img_bp)
app.register_blueprint(video_bp)
app.register_blueprint(audio_bp)
app.register_blueprint(forgery_bp)

if __name__ == '__main__':
    #app.run(debug=True,use_reloader=False)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

