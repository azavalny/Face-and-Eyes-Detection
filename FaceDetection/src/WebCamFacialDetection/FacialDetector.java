package WebCamFacialDetection;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

/***
 * Through a webcam input, the FacialDetector object detects and displays the face and eyes on a GUI
 * @author Alex Zavalny
 *
 */
public class FacialDetector {

	//eyes and face detector classifiers, with their own locations
	private CascadeClassifier eyesClassifier;
	private String eyesClassifierLocation;
	
	private CascadeClassifier faceClassifier;
	private String faceClassifierLocation;
	
	//Stage where the detection will be displayed on
	Mat stage = new Mat();
	
	/**
	 * Constructor for the Facial Detector Class
	 * Copy/Paste the file paths for both face and eyes classifier, and add a 2nd slash \\ to every \
	 * 		E.g. C:\Users should be changed to C:\\Users
	 */
	public FacialDetector(String faceClassLoc, String eyesClassLoc) {
	
		this.faceClassifier = new CascadeClassifier(faceClassLoc);
		this.eyesClassifier = new CascadeClassifier(eyesClassLoc);
	}
	
	
	/**
	 * Preprocesses the webcam, detects faces, and places them in a list
	 * @param currentFrame is the current frame of the webcam
	 * @return a list of the faces detected
	 */
	private List detectFaces(Mat currentFrame) {	
		//converting the webcam's color space into an 8-bit color space that
		//OpenCV can use 
		Imgproc.cvtColor(currentFrame, stage, Imgproc.COLOR_BGR2GRAY);
		
		//converts the current frame to grayscale by stretching out 
		//its histogram function allowing the face to be captured more easily
		Imgproc.equalizeHist(stage, stage);
		
		//Create a matrix of faces, and use the face classifier to fill it with faces
		MatOfRect faces = new MatOfRect();
		faceClassifier.detectMultiScale(currentFrame, faces);
		
		//Transfers the Faces matrix into a List
		List<Rect> listOfFaces = faces.toList();
		return listOfFaces;
	}	
	

	/**
	 * Loops through each face in the list, in the currentFrame, and draws a rectangle around each face
	 * and eyes in each face
	 * 
	 * @param facesList is a list of faces to be displayed
	 */
	public void displayFaces(List<Rect> facesList, Mat currentFrame) {
		//for each face detected:
		for(Rect face : facesList) {
			//display a rectangle around the face, and center it
			Imgproc.rectangle(currentFrame, new Point(face.x, face.y), 
					new Point(face.x + face.width, face.y + face.height), new Scalar(0,0,255), 2);
			
			Mat faceROI = stage.submat(face); //<--For each face, we create a region of interest (ROI) so we can
																								//insert the eyes
			 displayEyes(faceROI, currentFrame, face);
			}	
		
	}
	
	/**
	 * In each face, we detect eyes using Region of Interest from a detected face
	 * @param faceROI (region of interest)
	 * @param currentFrame
	 * @param face
	 */
	private void displayEyes(Mat faceROI, Mat currentFrame, Rect face) {
		 MatOfRect eyes = new MatOfRect();
		 eyesClassifier.detectMultiScale(faceROI, eyes);
		 List<Rect> eyeList = eyes.toList();
		 
		 for(Rect eye: eyeList) {
			 Point eyeCenter = new Point(face.x + eye.x + eye.width/2, face.y + eye.y + eye.height/2);
			 int radius = (int) Math.round((eye.width + eye.height) * 0.25);
            Imgproc.circle(currentFrame, eyeCenter, radius, new Scalar(255, 0, 0), 4);
		 }
	}
	
	
	/**
	 * runs the facial detection system
	 */
	public void runFacialDetection() {
		//User's webcam
		VideoCapture webCam = new VideoCapture(0);
		
		if (!webCam.isOpened()) {
            System.err.println("Cannot reach webcam");
            System.exit(0);
        }
				
		//a frame of the webcam 
		Mat frame = new Mat();
		
		while(webCam.read(frame)) {
			if (frame.empty()) {
                System.err.println("Frames aren't being captured");
                break;
            }
			displayFaces(detectFaces(frame), frame);
			
			//resizes and displays the webcam
			Imgproc.resize(frame, frame, new Size(2050, 1500));
			 HighGui.imshow("Face Detection", frame);
				if (HighGui.waitKey(10) == 27) {
				System.err.println("Too much lag occured, try turning off other applications to free some more memory");
		        }
		}
	}
}
