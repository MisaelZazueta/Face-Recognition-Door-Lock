import warnings

warnings.filterwarnings("ignore")
import numpy as np
import cv2
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
from rPPG.rPPG.rPPG_lukas_Extracter import *
from rPPG.rPPG.rPPG_Extracter import *

# load YAML and create model
yaml_file = open(r"C:\Users\ingmi\PycharmProjects\FaceNet-rPPG\rPPG\trained_model\RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights(r"C:\Users\ingmi\PycharmProjects\FaceNet-rPPG\rPPG\trained_model\RGB_rPPG_merge_softmax_.h5")
print("[INFO] Model is loaded from disk")

dim = (128, 128)


def get_rppg_pred(frame):
    use_classifier = True  # Toggles skin classifier
    use_flow = False  # (Mixed_motion only) Toggles PPG detection with Lukas Kanade optical flow
    sub_roi = []  # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]
    use_resampling = True  # Set to true with webcam

    fftlength = 300
    fs = 20
    # f = np.linspace(0,fs/2,fftlength/2 + 1) * 60;

    timestamps = []
    time_start = [0]

    break_ = False

    rPPG_extracter = rPPG_Extracter()
    rPPG_extracter_lukas = rPPG_Lukas_Extracter()
    bpm = 0

    dt = time.time() - time_start[0]
    time_start[0] = time.time()
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[-1] + dt)

    rPPG = []

    rPPG_extracter.measure_rPPG(frame, use_classifier, sub_roi)
    rPPG = np.transpose(rPPG_extracter.rPPG)

    # Extract Pulse
    if rPPG.shape[1] > 10:
        if use_resampling:
            t = np.arange(0, timestamps[-1], 1 / fs)

            rPPG_resampled = np.zeros((3, t.shape[0]))
            for col in [0, 1, 2]:
                rPPG_resampled[col] = np.interp(t, timestamps, rPPG[col])
            rPPG = rPPG_resampled
        num_frames = rPPG.shape[1]

        t = np.arange(num_frames) / fs
    return rPPG


def make_pred(li):
    [single_img, rppg] = li
    single_img = cv2.resize(single_img, dim)
    single_x = img_to_array(single_img)
    single_x = np.expand_dims(single_x, axis=0)
    single_pred = model.predict([single_x, rppg])
    return single_pred


def liveness(frame, collected_results, counter, accepted_falses, frames_buffer, faces, faces_box):
    for (x, y, w, h) in faces:
        sub_img = frame[y:y + h, x:x + w]
        rppg_s = get_rppg_pred(sub_img)
        rppg_s = rppg_s.T
        pred = make_pred([sub_img, rppg_s])
        collected_results.append(np.argmax(pred))
        counter += 1
        cv2.putText(frame, "Real: " + str(pred[0][0]), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                    lineType=cv2.LINE_AA)
        cv2.putText(frame, "Fake: " + str(pred[0][1]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                    lineType=cv2.LINE_AA)
        if len(collected_results) == frames_buffer:
            if sum(collected_results) <= accepted_falses:
                faces_box.append((sub_img, (abs(x), abs(y)), (abs(x + w), abs(y + h))))
            else:
                faces_box.clear()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            collected_results.pop(0)
    return faces_box, frame
