from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
from forgery_part import forgery_bp
from swap_part import swap_bp
# from realtime_part import real_time_bp
from hinet_part import hinet_bp
from makeup_part import makeup_bp
from flask_socketio import SocketIO
from init import socketio
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


app = Flask(__name__)
socketio.init_app(app)

app.register_blueprint(forgery_bp)
app.register_blueprint(swap_bp)
# app.register_blueprint(real_time_bp)
app.register_blueprint(hinet_bp)
app.register_blueprint(makeup_bp)


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/forgery')
def forgery():
    return render_template('forgery.html')

@app.route('/anonymization')
def anonymization():
    return render_template('anonymization.html')

@app.route('/makeup')
def makeup():
    return render_template('makeup.html')

@app.route('/tamper')
def tamper():
    return render_template('tamper.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')


if __name__ == '__main__':
    # app.run(port=8081,debug=True)
    socketio.run(app, host='0.0.0.0', port=8080, ssl_context=('/home/chenyidou/x_test/web/cert.pem', '/home/chenyidou/x_test/web/key.pem'),debug=True)
