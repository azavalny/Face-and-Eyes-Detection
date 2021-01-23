package WebCamFacialDetection;

import org.opencv.core.Core;

public class FacialDetectionDemo {

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		FacialDetector test = new FacialDetector("C:\\Users\\Name\\Desktop\\FaceDetection\\cascade files\\haarcascade_frontalface_alt.xml", 
												"C:\\Users\\Name\\Desktop\\FaceDetection\\cascade fileshaarcascade_eye_tree_eyeglasses.xml");
		test.runFacialDetection();
	}
}
