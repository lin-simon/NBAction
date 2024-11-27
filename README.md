# NBAction
Basketball is a high-paced game requiring precision, coordination and quick decision making. Understanding player actions is crucial for analysis and enjoyment of the sport. There are few systems available for live analysis of actions within dynamic sports environments which are common in basketball games. We present NBAction, a real-time basketball action classification and detection system utilizing computer vision techniques and frameworks. NBAction classifies common basketball actions such as shooting, scoring and defending, while also keeping track of the players, basketball and the location of the net. Our system employs a combination of self-trained and preexisting deep learning models for object recognition and classification to ensure high accuracy in a variety of basketball scenarios.
![NBAction](https://github.com/lin-simon/NBAction/blob/main/assets/ui.png?raw=true)

# Prerequisites
- Python 3.8 or higher and pip installed on your system
- [Optional] GPU-enabled environment for faster performance

# Installation Guide
- Clone this repository onto your personal environment through Git.
```
git clone https://github.com/lin-simon/NBAction.git
cd NBAction
```
- Alternatively, you can just download the zip and extract into your editor of choice.
- Next, install all dependencies, you can find a list of these at ```NBAction/requirements.txt```
```
pip install -r requirements.txt
or
pip install ultralytics opencv-python numpy     (ultralytics should automatically install any missing dependencies)
```
Once fully installed, navigate to nbaction.py.

self.video = cv2.VideoCapture("testset/VIDEO_FILENAME_HERE") -- In this line, replace the placeholder with a filepath to the video you wish to test, for ease of use, put it into the testset folder which contains a number of our test videos already.

For example, "testset/YourVideo.mov", to test it on "YourVideo.mov"

Then, run the following into your command line:
```
python nbaction.py 
```
make sure you are in the correct directory to run this, otherwise just run it through your IDE of choice.

Depending on where on your machine it is installed, you may have to change filepaths in nbaction.py to load any of your own videos into the program. ```/testset``` already contains some videos you can try out to see how NBAction works.

Enjoy!


# Features
- Real-time Object Detection: Identifies basketballs, hoops, players, and shooting actions in recorded or live video footage.
- Shot Detection: Tracks basketball trajectories and detects successful scores.
- Stabilization: Ensures smooth tracking of hoops and balls across frames to minimize anomalies.
- Scoring Visualization: Displays scoring information with overlays and effects.

# Training and Validation Loss Metrics
![Results](https://github.com/lin-simon/NBAction/blob/main/assets/results.png?raw=true)
