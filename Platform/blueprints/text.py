# 主页/文本检测视图
from flask import Blueprint, render_template, request
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks



bp = Blueprint("text", __name__, url_prefix="/")

# 初始化情感分类模型
semantic_cls = pipeline(Tasks.text_classification, model='model/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')

# http://127.0.0.1:5000
@bp.route("/", methods=['GET', 'POST'])
def index():
    result = None
    text = ""
    if request.method == 'POST':
        text = request.form['text-input']

        raw_result = semantic_cls(input=text)
        scores = raw_result['scores']
        labels = raw_result['labels']

        result_list = [f"{label}：{score * 100: .2f}%" for label, score in zip(labels, scores)]

        result = "、".join(result_list)

    return render_template('index.html', result=result, text=text)

