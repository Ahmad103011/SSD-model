import torch
import torchvision
from torchvision import transforms as T
import cv2
import cvzone

# Load the pretrained SSD model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

# Load class names from file
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Define transformation
imgtransform = T.ToTensor()

# Initialize video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Running real-time object detection...")
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Copy and preprocess the frame
    img = frame.copy()
    image = imgtransform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    with torch.no_grad():
        ypred = model(image)

    # Process predictions
    bbox, scores, labels = ypred[0]['boxes'], ypred[0]['scores'], ypred[0]['labels']
    nums = torch.argwhere(scores > 0.80).shape[0]  # Confidence threshold
    for i in range(nums):
        x, y, w, h = bbox[i].cpu().numpy().astype('int')
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 5)
        classname = labels[i].cpu().numpy().astype('int')
        classdetected = classnames[classname - 1]
        cvzone.putTextRect(img, classdetected, [x, y + 100], scale=2, border=2)

    # Display the frame with detections
    cv2.imshow('Real-Time Detection', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
