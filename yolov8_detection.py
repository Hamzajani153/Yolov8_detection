from ultralytics import YOLO
import cv2
import os


cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

output_dir = "./crop_images/"
frame_count = 0

while cap.isOpened():
    ret , frame = cap.read()
    results = model(frame,stream=True)
    print(model.names)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # print(box) 
            detection= box.data.tolist()     

            for i in detection:
                cls = int(box.cls[0])
                # print(classNames[cls])
                conf = i[4]
            
            x,y,w,h = box.xyxy[0]
            x,y,w,h = int(x) , int(y) , int(w), int(h)
            cropped_image = frame[y:h, x:w, :]
            print(cropped_image)
            cv2.rectangle(frame , (x,y),(w,h),(255,0,255), 3)
            cv2.putText(frame, f"Confidence: {conf:.2f}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            
            cv2.imwrite(os.path.join(output_dir, f"cropped_image_{frame_count}.jpg"), cropped_image)

            frame_count += 1
            
    cv2.imshow("frame" , frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()