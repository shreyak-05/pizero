import cv2

# Use default webcam (inex 0)
camera_index = 4

# Open webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Display the frame
    cv2.imshow("Webcam Feed", frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
