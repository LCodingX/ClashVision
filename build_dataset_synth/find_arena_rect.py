import cv2
from pathlib import Path
global height, width
def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x/width, y/height)
        print(x, y)
        

img_path = Path("~/dev/KataCR/Clash-Royale-Detection-Dataset/frames/gameplay36_tag/frame_0079.png").expanduser()
image = cv2.imread(str(img_path))
height, width = image.shape[:2]
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
print(height, width)
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1)
    if key == ord("q"): break

cv2.destroyAllWindows()