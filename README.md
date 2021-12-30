# Vehicle-Detection-Tracking-and-Distance-Measurement-on-the-Road
1. Used the YOLOv3 model to detect vehicles in the video and give each vehicle a unique ID
2. Improved the detection accuracy by embedding the Sequeeze-and-Excitation block (SE block) into the detection model
3. Applied Deep SORT to track detected vehicles
4. calculated the distance between the detected vehicle and default by using the principle of similar triangles and calibrating the camera's focal length and other parameters
5. Designed a GUI, which includes the button to import the video, start the recognition, and display the ID and location of the tracked vehicle in real time, and the imported video, and applied PyQt5 to achieve
