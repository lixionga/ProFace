from flask import Flask, request, jsonify, send_from_directory,Blueprint
import os
import cv2
from Read import load_embs_features
import time
from hinet_test import Hinet_model
hinet_bp = Blueprint('hinet_bp',__name__)

UPLOAD_FOLDER = '/home/chenyidou/x_test/web/uploads'
PROCESSED_FOLDER = 'static/out/hinet'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

hinet = Hinet_model()

@hinet_bp.route('/fangyu',methods=['POST'])
def hinet_for():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({'error': '未找到文件'}), 400
    timestamp = int(time.time()) % 1000
    old_filename = uploaded_file.filename
    new_filename = str(timestamp).zfill(3) + '_' + old_filename
    secret_filename = str(timestamp) + '_secret' + old_filename
    file_path = os.path.join(UPLOAD_FOLDER, new_filename)
    uploaded_file.save(file_path)
    # filename = new_filename.lower()

    result,secret = hinet.img_make(file_path)
    out_path = os.path.join(PROCESSED_FOLDER, new_filename)
    secret_path = os.path.join(PROCESSED_FOLDER, secret_filename)

    cv2.imwrite(out_path, result)
    cv2.imwrite(secret_path, secret)
    return jsonify({'success': True,'watermark_url': secret_path,
        'defense_result_url': out_path}),200

@hinet_bp.route('/jiance',methods=['POST'])
def hinet_rev():
    # original_image = request.files.get('original_image')
    detect_image = request.files.get('detect_image')
    # original_name = original_image.filename
    detect_name = detect_image.filename
    # file_path = os.path.join(UPLOAD_FOLDER, original_name)
    file_path1 = os.path.join(UPLOAD_FOLDER, detect_name)
    # original_image.save(file_path)
    detect_image.save(file_path1)
    result,real_mask,check_mask = hinet.img_check(file_path1)
    timestamp = int(time.time())
    timestamp = int(time.time()) % 1000

    # original_name = str(timestamp).zfill(3) + '_' + original_name
    detect_name = str(timestamp).zfill(3) + '_' + detect_name

    original_path = os.path.join(PROCESSED_FOLDER, 'real_mask.jpg')
    detect_path = os.path.join(PROCESSED_FOLDER, detect_name)
    cv2.imwrite(original_path, real_mask)
    cv2.imwrite(detect_path, check_mask)
    out = 1 - result[1]
    if out > 0.5:
        result_text = "已篡改"
    else:
        result_text = "未篡改"
    percentage = f"{out * 100 :.2f}"
    # result_text = "篡改"
    # percentage = '98.3%'
    img_path = 'static/out/hinet/367_340_6.jpg'
    det_path = 'static/out/hinet/146_340_6.jpg'
    return jsonify({
        'success':True,
        'resultText':result_text,
        'fakeProbability':percentage,
        'originalWatermarkUrl':original_path,
        'detectedWatermarkUrl':detect_path
    })


@hinet_bp.route('/processed/<path:filename>',methods=['GET'])
def download(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@hinet_bp.route('/download/<path:filename>',methods=['GET'])
def download_origin(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)