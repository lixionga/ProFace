from flask import Blueprint, render_template, request

bp = Blueprint("image", __name__, url_prefix="/image", template_folder="templates")

# http://127.0.0.1:5000/image
@bp.route("/yellowphoto")
def image():
    return render_template('page_yellowphoto1.html')

@bp.route("/ad")
def ad():
    return render_template('page_ad.html')

@bp.route("/illegal")
def illegal():
    return render_template('page_Illegal_detection.html')

@bp.route("/sensitive")
def sensitive():
    return render_template('page_Sensitive_detection.html')

@bp.route("/terrorism")
def terrorism():
    return render_template('page_Terrorism_identification.html')

@bp.route("/abuse")
def abuse():
    return render_template('page_abuse.html')

@bp.route("/irrigation")
def irrigation():
    return render_template('page_Irrigation_identification.html')


