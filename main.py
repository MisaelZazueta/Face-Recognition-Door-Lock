import cv2, time
from FaceNet.utils import *
from FaceNet.Recognizer import recognize
from keras.models import load_model
from FaceDetection.FaceDetector import faceDetection
from rPPG.test_model import liveness
import serial

if __name__ == "__main__":
    encoder_model = r'C:\Users\ingmi\PycharmProjects\FaceNet-rPPG\FaceNet\model\facenet_keras.h5'
    encodings_path = r'C:\Users\ingmi\PycharmProjects\FaceNet-rPPG\FaceNet\encodings\encodings_4.pkl'
    face_detector = cv2.CascadeClassifier(r"C:\Users\ingmi\PycharmProjects\FaceNet-rPPG\haarcascade_frontalface_default.xml")
    face_encoder = load_model(encoder_model)
    encoding_dict = load_pickle(encodings_path)
    items = encoding_dict.items()
    vc = cv2.VideoCapture(0)
    time.sleep(2)
    collected_results = []
    counter = 0          # count collected buffers
    frames_buffer = 5    # how many frames to collect to check for
    accepted_falses = 1  # how many should have zeros to say it is real
    ser = serial.Serial('COM3', 9600)
    result = False

    while True:
        faces_box = []
        ret, frame = vc.read()
        if not ret:
            print("no frame:(")
            break
        try:
            faces = faceDetection(frame, face_detector)
            faces_box, frame = liveness(frame, collected_results, counter, accepted_falses, frames_buffer, faces, faces_box)
        except:
            print("...")
        if len(faces_box) > 0:
            frame, result = recognize(frame, face_encoder, items, faces_box)
        frame = cv2.resize(frame, (1820, 1024))
        cv2.imshow('Camera', frame)
        if result:
            ser.write(b'A')
            while True:
                response = ser.readline().decode().strip()  # Leer la respuesta de Arduino
                if response == "Accion":
                    result = False
                    time.sleep(2)
                    break  # Salir del bucle si se recibió la respuesta esperada
        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
    vc.release()
    cv2.destroyAllWindows()

