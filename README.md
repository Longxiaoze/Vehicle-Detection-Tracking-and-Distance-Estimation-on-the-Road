# Vehicle-Detection-Tracking-and-Distance-Measurement-on-the-Road
1. Used the YOLOv3 model to detect vehicles in the video and give each vehicle a unique ID
2. Improved the detection accuracy by embedding the Sequeeze-and-Excitation block (SE block) into the detection model
3. Applied Deep SORT to track detected vehicles
4. calculated the distance between the detected vehicle and default by using the principle of similar triangles and calibrating the camera's focal length and other parameters
5. Designed a GUI, which includes the button to import the video, start the recognition, and display the ID and location of the tracked vehicle in real time, and the imported video, and applied PyQt5 to achieve
<img width="1276" alt="Screen Shot 2021-12-30 at 1 53 22 PM" src="https://user-images.githubusercontent.com/93239143/147725591-7d553a18-a9d1-4cf4-9117-cd55ff90e3e7.png">
<img width="1280" alt="Screen Shot 2021-12-30 at 1 55 18 PM" src="https://user-images.githubusercontent.com/93239143/147725598-fc434a76-2010-4e72-a9dc-c7fad1fed643.png">
![WechatIMG21](https://user-images.githubusercontent.com/93239143/147844563-bee3a9f9-b95c-4ffa-8351-acbc65fd71b7.jpeg)
