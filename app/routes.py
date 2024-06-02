from flask import Blueprint, render_template, request, send_from_directory, Response
import cv2
import os
import time
from ultralytics import YOLO

main_bp = Blueprint('main', __name__)

@main_bp.route("/about")
def about():
    return render_template('about.html')

@main_bp.route("/")
def hello_world():
    return render_template('index.html')

@main_bp.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                model = YOLO('yolov9c.pt')
                detections = model(img, save=True)
            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                model = YOLO('yolov9c.pt')
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, save=True)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)
                    if cv2.waitKey(1) == ord('q'):
                        break
                cap.release()
                out.release()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    image_path = folder_path + '/' + latest_subfolder + '/' + f.filename
    return render_template('index.html', image_path=image_path)

@main_bp.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path+'/'+latest_subfolder
    files = os.listdir(directory)
    latest_file = files[0]

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ)
    else:
        return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)

@main_bp.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
