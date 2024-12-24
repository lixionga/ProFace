from flask import Flask, jsonify, Blueprint, render_template, request, json
import logging
from flask_sslify import SSLify
from video_clip.translate_baidu import zh2en
from video_clip.retrieval import sim_image_text
from video_clip.video_do import video_segment
from video_clip.visua_grounding import visual_grouding_do
from video_clip.retrieval_zh import sim_image_text_zh
import threading
import time
import os

bp = Blueprint("video", __name__, url_prefix="/video")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 获取根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 获取result路径
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'result')  # 服务器上传路径

def clean_files(static_path=None, model_path=None, heatmap_path=None):
    #  清理上次推理的结果
    logger.info("清理上次推理结果")
    try:
        paths_to_clean = []

        # 清理face_results目录
        if os.path.exists(RESULT_FOLDER):
            for file in os.listdir(RESULT_FOLDER):
                file_path = os.path.join(RESULT_FOLDER, file)
                if os.path.isfile(file_path):
                    paths_to_clean.append(file_path)

        # 执行文件删除
        for path in paths_to_clean:
            try:
                os.remove(path)
                logger.info(f"成功删除文件: {path}")
            except Exception as e:
                logger.error(f"删除文件失败 {path}: {str(e)}")

    except Exception as e:
        logger.error(f"清理文件时出错: {str(e)}")

def image_do():
    time.sleep(10)
    if os.path.exists('./static/video/file.mp4'):
        video_segment()

# http://127.0.0.1:5000/video
@bp.route("/video")
def video():
    logger.info("访问视频检索主页面")
    return render_template('page_video.html')

@bp.route('/upload', methods=['POST'])
def upload():
    logger.info("收到文件上传请求")
    clean_files()
    try:
        file = request.files['file']
        file.save('./static/video/file.mp4')

        image_olds = os.listdir('./static/img')
        for image_old in image_olds:
          os.remove(os.path.join('./static/img', image_old))
        return {'code': 0}
    except Exception as e:
        logger.error(f"处理上传文件时出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/text2i', methods=['POST'])
def text2i():
    if os.path.exists('./static/video/file.mp4'):
        video_segment()
    data = request.get_json()
    text_t = data['text']
    # text_en = zh2en(text_t)
    top_3 = sim_image_text_zh(text_t)
    pic_1 = visual_grouding_do(str(top_3[0]), text_t)
    pic_2 = visual_grouding_do(str(top_3[1]), text_t)
    pic_3 = visual_grouding_do(str(top_3[2]), text_t)
    pic_4 = visual_grouding_do(str(top_3[3]), text_t)
    text_number = [str(top_3[0]), str(top_3[1]), str(top_3[2]), str(top_3[3])]
    return jsonify(text_number)