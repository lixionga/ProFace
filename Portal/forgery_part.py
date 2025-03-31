from flask import Flask, request, jsonify, render_template, send_from_directory, url_for,Blueprint
import os
from Mobile_Faceswap import Mobile_face
import cv2
from Read import load_embs_features
import time
from forgery_infere import ForgeryInference

# app = Flask(__name__)

forgery_bp = Blueprint('forgery_bp', __name__)

CAM_OUTPUT_FOLDER = '/home/chenyidou/x_test/web/static/out/cap_out'
os.makedirs(CAM_OUTPUT_FOLDER, exist_ok=True)
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# forgery_bp.config['CAM_OUTPUT_FOLDER'] = CAM_OUTPUT_FOLDER



@forgery_bp.route('/upload2', methods=['POST'])
def upload_frogery():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({'error': '未找到文件'}), 400
    timestamp = int(time.time()) % 1000
    old_filename = uploaded_file.filename
    new_filename = str(timestamp) + '_' + old_filename

    file_path = os.path.join(UPLOAD_FOLDER,new_filename)
    # print(file_path)
    uploaded_file.save(file_path)


    model = ForgeryInference()
    result,image_path = model.forward(file_path)
    
    file_name = os.path.basename(image_path).split('.')[0] + '.jpg'
    img_path = os.path.join('static/out',file_name)
    return jsonify({
        'message': '上传并处理成功',
        'heatmap_path': img_path,
        'model_score': result
    }), 200

@forgery_bp.route('/cam_output/<path:filename>', methods=['GET'])
def get_cam_output_file(filename):
    print('get_cam_output_file:', filename)
    return send_from_directory(CAM_OUTPUT_FOLDER, filename)

@forgery_bp.route('/download_cam/<path:filename>', methods=['GET'])
def download_cam_output_file(filename):
    return send_from_directory(CAM_OUTPUT_FOLDER ,filename, as_attachment=True)