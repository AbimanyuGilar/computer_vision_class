import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    img = detector.findPose(img)

    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
    
    if lmList:
        length, img, info = detector.findDistance(
            lmList[11][0:2],
            lmList[15][0:2],
            img=img,
            color=(255,0,0),
            scale=10
        )

        angle, img = detector.findAngle(
            lmList[11][0:2],
            lmList[13][0:2],
            lmList[15][0:2],
            img=img,
            color=(0,0,255),
            scale=10
        )

        print("Jarak: ", length)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q') : break

cap.release()
cv2.destroyAllWindows()

