from flask import Blueprint, render_template, request, send_file, send_from_directory, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
import json
import logging
import time
import shutil
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forgery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

bp = Blueprint("forgery", __name__, url_prefix="/forgery")

# 获取根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 使用绝对路径定义所有路径
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'images', 'uploads')  # 服务器上传路径
MODEL_UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'model', 'EG', 'pictures_folder')  # 图片上传模型路径
CROPPED_FOLDER = os.path.join(MODEL_UPLOAD_FOLDER, 'cropped_results', 'cropped_faces')  #  图片裁剪输出路径
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'model', 'EG', 'cam_output')  # 热力图输出路径
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'face_results')  # 压缩结果输出路径
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'model', 'EG', 'weights', 'EG_FF++(raw).tar')  # 模型权重路径
SCRIPT_DIR = os.path.join(PROJECT_ROOT, 'model', 'EG')  # 脚本路径
SCRIPT_PATH = os.path.join(SCRIPT_DIR, 'inference_image.py')  # 脚本执行

# 检查目录（不存在则创建）
for directory in [UPLOAD_FOLDER, MODEL_UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(directory, exist_ok=True)

def allowed_file(filename):
    # 指定图片格式
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_inference(image_path):
    # 运行推理脚本并获取结果
    try:
        logger.info(f"开始处理图片: {image_path}")
        
        # 确保所有路径都是绝对路径
        abs_image_path = os.path.abspath(image_path)
        abs_weights_path = os.path.abspath(WEIGHTS_PATH)
        abs_output_dir = os.path.abspath(OUTPUT_FOLDER)
        
        # 检查文件是否存在
        if not os.path.exists(abs_image_path):
            logger.error(f"输入图片不存在: {abs_image_path}")
            return None
            
        if not os.path.exists(abs_weights_path):
            logger.error(f"模型权重文件不存在: {abs_weights_path}")
            return None
        
        # 构建命令
        cmd = [
            'python',
            SCRIPT_PATH,
            '-w', abs_weights_path,
            '-input_image', abs_image_path,
            '--device', 'cpu',
            '--method', 'gradcam++',
            '--output_dir', abs_output_dir
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,  # 捕获输出
            stderr=subprocess.PIPE,  # 捕获错误
            text=True,  # 指定输出为文本
            env=os.environ.copy()  # 使用当前环境变量
        )
        
        # 等待输出结果
        stdout, stderr = process.communicate()
        
        logger.info(f"命令标准输出: {stdout}")
        if stderr:
            logger.error(f"命令错误输出: {stderr}")
        
        # 如果子程序执行出错
        if process.returncode != 0:
            logger.error(f"命令执行失败，e: {process.returncode}")
            if "No face is detected" in stderr:
                return 0.0  # 检测失败，返回伪造概率为0
            return None
        
        # 从输出中提取fakeness值
        for line in stdout.split('\n'):
            if 'fakeness:' in line:
                try:
                    fakeness = float(line.split(':')[1].strip())
                    logger.info(f"分析结果 fakeness: {fakeness}")
                    # 保存为txt文件
                    save_fakeness_to_file(fakeness, abs_image_path)
                    return fakeness
                except ValueError as e:
                    logger.error(f"获取fakeness值失败: {e}")
                    return None
                
        logger.error("未找到fakeness值")
        return None
        
    except Exception as e:
        logger.error(f"执行分析过程出错: {str(e)}", exc_info=True)
        return None

def save_fakeness_to_file(fakeness, image_path):
    #  将fakeness值保存为txt
    try:
        # 确保 UPLOAD_FOLDER 路径存在
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        # 获取图片文件名
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 生成txt文件路径
        txt_filename = f"{image_name}_fakeness.txt"
        txt_path = os.path.join(UPLOAD_FOLDER, txt_filename)

        # 写入fakeness
        fakeness_percentage = f"{fakeness * 100:.2f}%"
        with open(txt_path, 'w') as f:
            f.write(f"Fakeness: {fakeness_percentage}\n")

        logger.info(f"Fakeness值已保存到文件: {txt_path}")
    except Exception as e:
        logger.error(f"保存Fakeness值出错: {str(e)}", exc_info=True)


def clean_files(static_path=None, model_path=None, heatmap_path=None):
    #  清理文件上传目录以输入新的文件
    try:
        paths_to_clean = []
        
        # 清理static/images/images/uploads目录
        if os.path.exists(UPLOAD_FOLDER):
            for file in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, file)
                if os.path.isfile(file_path):
                    paths_to_clean.append(file_path)
        
        # 清理model/EG/pictures_folder目录
        if os.path.exists(MODEL_UPLOAD_FOLDER):
            for file in os.listdir(MODEL_UPLOAD_FOLDER):
                file_path = os.path.join(MODEL_UPLOAD_FOLDER, file)
                if os.path.isfile(file_path):
                    paths_to_clean.append(file_path)

        # 清理model/EG/pictures_folder/cropped_faces目录
        if os.path.exists(CROPPED_FOLDER):
            for file in os.listdir(CROPPED_FOLDER):
                file_path = os.path.join(CROPPED_FOLDER, file)
                if os.path.isfile(file_path):
                    paths_to_clean.append(file_path)

        # 清理cam_output目录
        if os.path.exists(OUTPUT_FOLDER):
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    paths_to_clean.append(file_path)

        # 清理face_results目录
        if os.path.exists(RESULTS_FOLDER):
            for file in os.listdir(RESULTS_FOLDER):
                file_path = os.path.join(RESULTS_FOLDER, file)
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

@bp.route("/forgery")
def forgery_page():
    logger.info("访问伪造检测主页面")
    return render_template('base_forgery.html')

@bp.route("/forgery/upload", methods=['POST'])
def upload_file():
    logger.info("收到文件上传请求")
    
    # 清理上一次上传的图片
    clean_files()

    # 检测是否存在文件或者URL
    file = request.files.get('file')
    image_url = request.form.get('image_url', '').strip()
    
    if not file and not image_url:
        logger.error("未提供文件或URL")
        return jsonify({'error': 'No file or URL provided'}), 400

    try:

        unique_filename = None

        # 文件上传模式
        if file:
            if file.filename == '':
                logger.error("没有选择文件")
                return jsonify({'error': 'No selected file'}), 400

            if not allowed_file(file.filename):
                logger.error(f"不支持的文件类型: {file.filename}")
                return jsonify({'error': 'Invalid file type'}), 400

            # 生成上传文件名，使用时间戳
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{filename}"
            
            # 保存到static目录
            static_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            logger.info(f"保存图片到static目录: {static_path}")
            file.save(static_path)

        else:
            # URL上传模式
            logger.info(f"图片URL为：{image_url}")
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code != 200:
                    logger.error(f"无法访问URL：{image_url}, {response.status_code}")
                    return jsonify({'error': 'Failed to download image'}), 400

                if 'image' not in response.headers.get('Content-Type', ''):
                    logger.error(f"URL内容不是图片: {image_url}")
                    return jsonify({'error': 'URL does not link to image'}), 400

                # 保存下载的图片
                timestamp = int(time.time())
                filename = f"{timestamp}_download_image.jpg"
                static_path = os.path.join(UPLOAD_FOLDER, filename)
                with open(static_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"图片已下载到：{static_path}")
                unique_filename = filename
            except requests.exceptions.RequestException as e:
                logger.error(f"请求图片URL时出错：{str(e)}")
                return jsonify({'error': 'Failed to download image'}), 500

        # 保存到上传目录
        model_path = os.path.join(MODEL_UPLOAD_FOLDER, unique_filename)
        logger.info(f"复制图片到上传目录: {model_path}")
        shutil.copy2(static_path, model_path)

        # 运行推理
        logger.info("开始分析————————————————————————————————————————————————————————————————————")
        fakeness = run_inference(model_path)

        if fakeness is None:
            logger.error("分析出错")
            clean_files()  # 执行文件清理
            return jsonify({'error': 'Inference failed'}), 500

        # 查找热力图文件并复制到static目录
        logger.info("查找热力图及裁剪结果")
        heatmap_filename = None
        heatmap_static_path = None
        cropped_filename = None
        cropped_static_path = None

        for f in os.listdir(OUTPUT_FOLDER):
            # 检测是否存在热力图
            if f.endswith(('_gradcam++_cam.jpg', '_gradcam++_cam.png')):
                heatmap_filename = f
                logger.info(f"找到热力图文件: {f}")
                # 复制热力图到static目录
                static_heatmap_filename = f"heatmap_{timestamp}_{f}"
                heatmap_static_path = os.path.join(UPLOAD_FOLDER, static_heatmap_filename)
                heatmap_original_path = os.path.join(OUTPUT_FOLDER, f)
                try:
                    shutil.copy2(heatmap_original_path, heatmap_static_path)
                    logger.info(f"复制热力图到: {heatmap_static_path}")
                except Exception as e:
                    logger.error(f"复制热力图失败: {str(e)}")
                    continue
                break

        if heatmap_filename:
            heatmap_url = f'/static/images/uploads/heatmap_{timestamp}_{heatmap_filename}'
            logger.info(f"热力图路径: {heatmap_url}")
        else:
            logger.error("未找到热力图文件")
            heatmap_url = None

        for f in os.listdir(CROPPED_FOLDER):
            # 检测是否存在裁剪结果
            if f.endswith(('_cropped.jpg', '_cropped.png')):
                cropped_filename = f
                logger.info(f"找到裁剪文件: {f}")
                # 复制裁剪结果到static目录
                static_cropped_filename = f"cropped_{timestamp}_{f}"
                cropped_static_path = os.path.join(UPLOAD_FOLDER, static_cropped_filename)
                cropped_original_path = os.path.join(CROPPED_FOLDER, f)
                try:
                    shutil.copy2(cropped_original_path, cropped_static_path)
                    logger.info(f"复制裁剪结果到: {cropped_static_path}")
                except Exception as e:
                    logger.error(f"复制结果失败: {str(e)}")
                    continue
                break

        if cropped_filename:
            cropped_url = f'/static/images/uploads/cropped_{timestamp}_{heatmap_filename}'
            logger.info(f"裁剪结果路径: {cropped_url}")
        else:
            logger.error("未找到裁剪结果")
            heatmap_url = None

        result = {
            'fakeness': f"{fakeness:.4f}",
            'heatmap_path': heatmap_url,
            'original_image': f'/static/images/uploads/{unique_filename}'
        }

        logger.info(f"返回结果: {result}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"处理上传文件时出错: {str(e)}", exc_info=True)
        clean_files()
        return jsonify({'error': str(e)}), 500



@bp.route('/cam_output/<filename>')
def cam_output_file(filename):
    logger.info(f"请求cam_output文件: {filename}")
    return send_from_directory(OUTPUT_FOLDER, filename)

@bp.route('/download', methods=['POST'])
def download_file():
    logger.info("收到文件下载请求")
    try:
        # 检测源文件是否存在
        if not os.path.exists(UPLOAD_FOLDER):
            logger.error(f"源文件不存在：{UPLOAD_FOLDER}")
            return None

        # 检测目标文件夹是否存在
        if not os.path.exists(RESULTS_FOLDER):
            os.makedirs(RESULTS_FOLDER)

        for f in os.listdir(UPLOAD_FOLDER):
            # 检测是否存在裁剪结果
            if f.endswith(('_cropped.jpg', '_cropped.png')):
                logger.info(f"找到结果文件: {f}")

        # 动态生成压缩包文件名
        timestamp = int(time.time())
        zip_filename = f"{timestamp}_results.zip"
        zip_path = os.path.join(RESULTS_FOLDER, zip_filename)

        # 检查路径
        logger.info(f"zip path: {zip_path}")

        # 压缩文件夹
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', UPLOAD_FOLDER)
        logging.info(f"生成zip文件: {zip_path}")
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename
        )
    except Exception as e:
        # 捕获异常并返回错误
        print(f"下载文件时发生错误： {str(e)}")
        return f"An error occurred: {str(e)}", 500