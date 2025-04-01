from flask import Flask, request, jsonify, render_template, send_from_directory, url_for,Blueprint
import os
# from Mobile_Faceswap import Mobile_face
# from Makeup-privacy.Makeup import MakeupPrivacy
import cv2
from Read import load_embs_features
import time
from Makeup import MakeupPrivacy

makeup_bp = Blueprint('makeup_bp', __name__)

UPLOAD_FOLDER = '/home/chenyidou/x_test/web/uploads'
PROCESSED_FOLDER = 'static/out/makeup'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
model = MakeupPrivacy()
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@makeup_bp.route('/make_upload', methods=['POST'])
def makeup():
    # if 'original_image' not in request.files or 'makeup_image' not in request.files:
    #     return jsonify({"success": False, "message": "请上传原图和妆容参考图"}), 400

    original_image = request.files['original_image']
    makeup_image = request.files['makeup_image']

    if original_image.filename == '' or makeup_image.filename == '':
        return jsonify({"success": False, "message": "文件未选择"}), 400

    if not allowed_file(original_image.filename) or not allowed_file(makeup_image.filename):
        return jsonify({"success": False, "message": "不支持的文件类型"}), 400

    original_image = request.files['original_image']

    makeup_image = request.files['makeup_image']
    timestamp = int(time.time()) % 1000

    old_filename1 = original_image.filename
    new_filename1 = str(timestamp) + '_' + old_filename1
    file_path1 = os.path.join(UPLOAD_FOLDER, new_filename1)
    original_image.save(file_path1)

    old_filename2 = makeup_image.filename
    new_filename2 = str(timestamp) + '_' + old_filename2
    file_path2 = os.path.join(UPLOAD_FOLDER, new_filename2)
    makeup_image.save(file_path2)

    out = model.forward(file_path1,file_path2)
    out_path = os.path.join(PROCESSED_FOLDER, new_filename1)
    cv2.imwrite(out_path, out)
    out_path1 = os.path.join(PROCESSED_FOLDER,old_filename1.split('.')[0] + '_' + old_filename2.split('.')[0] + '.jpg')
    return jsonify({"success": True, "processed_image_url": out_path1})

@makeup_bp.route('/processed/<path:filename>', methods=['GET'])
def get_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@makeup_bp.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)

    