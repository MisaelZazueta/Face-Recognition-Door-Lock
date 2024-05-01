from scipy.spatial.distance import cosine
from FaceNet.utils import *
import cv2

def recognize(img,
              encoder,
              items,
              boxes,
              recognition_t=0.25,
              confidence_t=0.99,
              required_size=(160, 160)):
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #results = detector.detect_faces(img_rgb)
    #results = detector.detectMultiScale(img_rgb)

    results = boxes
    for res in results:
        face, pt_1, pt_2 = res
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'Unknown'

        distance = float("inf")

        for db_name, db_encode in items:
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist


        if name == 'Unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            result = False
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
            result = True

    return img, result

