import cv2 as cv
import mediapipe as mp
import time


class handTracking:

    def __init__(self):

        self.myHands = mp.solutions.hands

        self.mydraw = mp.solutions.drawing_utils

        self.hands = self.myHands.Hands()

    def findHands(self, frame, draw=True):

        ImageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        self.result = self.hands.process(ImageRGB)

        if self.result.multi_hand_landmarks:
            for vansh_hand in self.result.multi_hand_landmarks:
                if draw:
                    self.mydraw.draw_landmarks(
                        frame, vansh_hand, self.myHands.HAND_CONNECTIONS)

        return frame

    def handsPosition(self, frame, handNumber=0, draw=True):
        self.lmList = []

        if self.result.multi_hand_landmarks:

            hand = self.result.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmList.append([id, cx, cy])
            # print(id, cx, cy)

                # if draw:
                #     cv.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

        print(self.lmList)

        return self.lmList
    
    def fingerUp(self, frame):
        fingers = []
        tipId = [4, 8, 12, 16, 20]
        
        if self.lmList[tipId[0]][1] < self.lmList[tipId[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0) 
        for id in range(1, 5):
            if self.lmList[tipId[id]][2] < self.lmList[tipId[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():

    img = cv.VideoCapture(0)

    pTime = 0

    cTime = 0

    obj = handTracking()

    while True:

        _, frame = img.read()

        frame = obj.findHands(frame, True)
        position = obj.handsPosition(frame, 0, True)

        if len(position) != 0:
            print(position[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)

        cv.imshow("video", frame)

        if cv.waitKey(1) == ord("q"):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
