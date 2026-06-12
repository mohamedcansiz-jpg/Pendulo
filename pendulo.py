import cv2
import numpy as np
import pandas as pd

video = "pendulo.mp4"

cap = cv2.VideoCapture(video)

fps = cap.get(cv2.CAP_PROP_FPS)

dado = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Faixa de laranja
    lower = np.array([5, 100, 100])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)

        if M["m00"] != 0:

            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])

            tempo = frame_num / fps

            dados.append([frame_num, tempo, x, y])

            cv2.circle(frame, (x, y), 5, (0,255,0), -1)
    frame = cv2.resize(frame, (540, 960))
    cv2.imshow("Pendulo", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(
    dados,
    columns=["frame", "tempo", "x", "y"]
)

df.to_csv("dados_pendulo.csv", index=False)
