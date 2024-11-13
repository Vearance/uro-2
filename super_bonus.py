import cv2
import streamlit as st
from ultralytics import YOLO

def app():
    st.title("Object Detection")

    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())  # Classes from pretrained YOLO

    select_objects = st.multiselect(
        'Choose objects to detect', object_names, default=['person']
    )

    min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5, step=0.1)

    start = st.button("Start webcam")
    if start:  # If True
        cap = cv2.VideoCapture(0)

        st.info("Starting webcam.")
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()  # Read a frame from webcam
            if not ret:  # If fails, show error and stop
                st.error("Failed to capture video.")
                break

            result = model(frame)  # Return array of detected objects in the frame
            for detection in result[0].boxes.data:  # Loop every detected object in the frame
                x0, y0 = int(detection[0]), int(detection[1])  # Coordinates of the box
                x1, y1 = int(detection[2]), int(detection[3])

                score = round(float(detection[4]), 2)

                detect_class = int(detection[5])
                object_name = model.names[detect_class]

                label = f"{object_name} {score}"

                if object_name in select_objects and score >= min_confidence:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x0, y0 - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

            st_frame.image(frame, channels="BGR")  # Show the frame

        cap.release()
        cv2.destroyAllWindows()
        st.success("Webcam stopped.")

if __name__ == "__main__":
    app()
