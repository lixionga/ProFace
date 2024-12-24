from flask import Blueprint, render_template, request

bp = Blueprint("audio", __name__, url_prefix="/audio")

# http://127.0.0.1:5000/audio/audio
@bp.route("/audio")
def audio():
    return render_template('page_music.html')