from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
from Mobile_Faceswap import Mobile_face
import cv2
from Read import load_embs_features
import time
from forgery_infere import ForgeryInference
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

CAM_OUTPUT_FOLDER = '/Users/mac/代码/web/cam_output'
os.makedirs(CAM_OUTPUT_FOLDER, exist_ok=True)
app.config['CAM_OUTPUT_FOLDER'] = CAM_OUTPUT_FOLDER

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER





# sourece_image = '/Users/mac/代码/web/image/0027_01.jpg'
# sourece_image1 = '/Users/mac/代码/web/image/0007_01.jpg'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forgery')
def forgery1():
    return render_template('forgery.html')

@app.route('/bitstring', methods=['POST'])
def receive_bitstring():
    global faceswap
    bit_array = request.json
    # if not isinstance(bit_array, list) or len(bit_array) != 16 or not all(bit in [0, 1] for bit in bit_array):
    #     return jsonify({'error': '无效的数组'}), 400
    if not isinstance(bit_array, list) or not all(bit in [0, 1] for bit in bit_array):
        return jsonify({'error': '无效的数组'}), 400
    # 处理接收到的 bit_array
    print('接收到的数组:', bit_array)
    id_emb_list,id_feature_list = load_embs_features()
    selected_embs = [id_emb_list[i] for i in range(len(bit_array)) if bit_array[i] == 1]
    selected_feats = [id_feature_list[i] for i in range(len(bit_array)) if bit_array[i] == 1]
    id_emb = 0
    id_feature = 0
    for emb in selected_embs:
        id_emb += emb
    for feature in selected_feats:
        id_feature += feature
    id_emb /= len(selected_embs)
    id_feature /= len(selected_feats)
    faceswap = Mobile_face(id_emb, id_feature)
    return jsonify({'message': '数组接收成功'})

@app.route('/upload', methods=['POST'])
def upload_and_process():
    global faceswap
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({'error': '未找到文件'}), 400

    timestamp = int(time.time())
    old_filename = uploaded_file.filename
    new_filename = str(timestamp) + '_' + old_filename

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    # print(file_path)
    uploaded_file.save(file_path)

    print('上传文件路径：', file_path)
    filename = new_filename.lower()
    print('文件名：', filename)
    if filename.endswith(('.jpg','.jpeg','.png')):

        result = faceswap.img_swap(file_path)
        cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], "processed_" + new_filename), result)
    elif filename.endswith(('.mp4','.avi','.mov')):
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(app.config['PROCESSED_FOLDER'], "processed_" + new_filename), fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                result = faceswap.frame_swap(frame)
            except:
                result = frame
            out.write(result)
        cap.release()
        out.release()
        print('视频处理完成')
    else:
        print('文件格式不支持')
        return jsonify({'error': '文件格式不支持'}), 400

    return jsonify({
        'message': '上传并处理成功',
        'processed_file': "processed_" + new_filename
    }), 200

@app.route('/upload2', methods=['POST'])
def upload_frogery():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({'error': '未找到文件'}), 400
    timestamp = int(time.time())
    old_filename = uploaded_file.filename
    new_filename = str(timestamp) + '_' + old_filename

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    # print(file_path)
    uploaded_file.save(file_path)


    model = ForgeryInference()
    result,image_path = model.forward(file_path)
    
    file_name = os.path.basename(image_path).split('.')[0] + '.jpg'
    return jsonify({
        'message': '上传并处理成功',
        'processed_file': file_name,
        'model_score': result
    }), 200



@app.route('/processed/<path:filename>', methods=['GET'])
def get_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/cam_output/<path:filename>', methods=['GET'])
def get_cam_output_file(filename):
    print('get_cam_output_file:', filename)
    return send_from_directory(app.config['CAM_OUTPUT_FOLDER'], filename)

@app.route('/download_cam/<path:filename>', methods=['GET'])
def download_cam_output_file(filename):
    return send_from_directory(app.config['CAM_OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
