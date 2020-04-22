package application;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import java.io.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;

import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import utilities.Utilities;

import java.util.ArrayList;


public class Controller {
	
	@FXML
	private ImageView imageView;
	private ScheduledExecutorService timer;
	// these are matrices for row and cols of STI and hist
	private Mat imageRow;
	private Mat imageCol; 
	private Mat histRow; 
	private Mat histCol;

	private boolean firstFrame; // determines if its our first frame
	private ArrayList<Mat> frames; 
	private Mat image; // our image
	private int frameCount; // total # of frame
	private static int frameNumber; // current frame #
	private int histogramBin; 
	private VideoCapture capture; 
	
	@FXML 
	private String grabFile() { // this function gets the file from the explorer
		FileChooser file = new FileChooser();
		
		File selectedFile = file.showOpenDialog(null);
		if(selectedFile != null) {
			String fileName = selectedFile.getAbsolutePath(); 
			return fileName;
		}
		else {
			System.out.println("Could not select file"); //file selection failed
			return null;
		}
	}
	@FXML
	protected void openImage(ActionEvent event) throws InterruptedException {
		final String imageFilename = grabFile(); //fetch filename
		String extension = imageFilename.substring(imageFilename.lastIndexOf("."),imageFilename.length());  //accesses the file type
		if ( extension.equals(".mov") || extension.equals(".wmv") || extension.equals(".mpeg") || extension.equals(".mp4") || extension.equals(".avi")) {
			capture = new VideoCapture(imageFilename);//open video file
			if(capture.isOpened()) {//open successfully
				firstFrame=true;
				createFrameGrabber();
				histSTI(null);
			}
			else
				System.out.println("Failed to open capture");
		}
		else
			System.out.println("Invalid type");
	}
	
	// These 4 functions will help display the row, col, and histogram STIs
	@FXML
	protected void R(ActionEvent event){
		imageView.setImage(Utilities.mat2Image(imageRow)); 
	}
	@FXML
	protected void C(ActionEvent event){
		imageView.setImage(Utilities.mat2Image(imageCol));
	}
	@FXML
	protected void HC(ActionEvent event) {
		imageView.setImage(Utilities.mat2Image(histRow)); 
	}
	@FXML
	protected void HR(ActionEvent event) {
		imageView.setImage(Utilities.mat2Image(histCol)); 
	}

	protected void createRowCol(ActionEvent event) { // create rows and cols for the STI (1.1)
		if (image != null) {
			// col operation
			Mat col = image.col((int) Math.ceil(image.cols()/2)); // get center col
			if(firstFrame == true) { // initialize mat if this is the first frame
				imageCol = new Mat (image.rows(),frameCount,CvType.CV_8UC3);
			}
			col.col(0).copyTo(imageCol.col(frameNumber)); //place col in appropriate STI col
			//same as above, but for rows
			Mat row = image.row((int) Math.ceil(image.rows()/2));
			if(firstFrame == true) {
				imageRow = new Mat (frameCount,image.cols(),CvType.CV_8UC3);
			}
			row.row(0).copyTo(imageRow.row(frameNumber));
			
			if(firstFrame == true) { // check if its our first frame
				firstFrame = false;
			}
		}
		else
			System.out.println("Image not selected");
		}
	
	protected void chromaticity(ActionEvent event) { // implements chromaticity feature (1.2)		
		if(frameNumber == 0) // create the arraylist if video has not started yet
			frames = new ArrayList<>();
		if (image != null) {
			image.convertTo(image, CvType.CV_64FC3);
			Mat varChromaticity = new Mat (image.rows(),image.cols(),CvType.CV_64FC2); 
			// convert RGB values from imageCol to array
			int sizeCol2 = (int) (image.total()*image.channels());
			double []  tempCol2 = new double [sizeCol2];
			image.get(0, 0,tempCol2);
			
			// Initialize array for storing values for the chromaticity matrix
			int sizeCol1 = (int)(varChromaticity.total()*varChromaticity.channels());
			double [] tempCol1 = new double [sizeCol1];
			
			// Begin the computation
			int ctr = 1; // keeps count of progression thru tempCol2
			int ctr2 = 0; // keeps count for when to put values in tempCol1
			for(int i = 0; i<sizeCol2; i++) {
				if (ctr%3==0) { // 3 because of the set {R, G, B} contains 3 elements
					double R = tempCol2 [i-2]; // fetch R value
					double G = tempCol2 [i-1]; // fetch G value
					double B = tempCol2 [i];   // fetch B value
					
					double R2, G2; // R and G variables for chromaticity
					if(R+G+B==0) { // Edge case for black
						R2= 0.00;
						G2 = 0.00;
					}
					else { // Chromaticity calc
						R2 = R / (R+G+B);
						G2 = G/(R+G+B);
					}
					if(ctr2 < sizeCol1) { // If ctr2 exceed its maximum value, put R2, G2 vars in array
						tempCol1[ctr2] = R2;
						ctr2++;
						tempCol1[ctr2] = G2;
						ctr2++;
					}
				}
				ctr++;
			}
			// Stick RG into the chromaticity matrix.
			varChromaticity.put(0, 0, tempCol1);
			// Re-convert to 3 channel matrix
			image.convertTo(image, CvType.CV_8UC3);
			frames.add(varChromaticity);
		}
		else
			System.out.println("Image not detected");
	}
	
	protected void histDiff(ArrayList<double[][]> hist , ArrayList<Double> varI) { // this is the I var from the formula (1.2)
		for(int i = 0 ; i < frameCount - 1 ; i++) { // Loop thru frames
			double[][] prev = hist.get(i);
			double[][] cur = hist.get(i+1);
			// Col hist STI
			double sumList = 0.0;
			for(int r = 0 ; r < histogramBin ; r++)
				for(int c = 0 ; c < histogramBin ; c ++)
					 sumList+= Math.min(prev[r][c], cur[r][c]);
			varI.add(sumList);
		}
	}
	
	protected void createHistogram(ArrayList<double[][]> Histogram , Mat varChromaticity , int n) { // Create histogram (1.2)
		histogramBin = (int)Math.floor(1+((double)Math.log(n)/(double)Math.log(2))); // Formula N = 1+log2(n)
		int R , G; // col and row for histogram
		double bounds = 1.00/histogramBin; // boundary for histogram bin
		// Make histogram for each frame
		int sz = (int) (varChromaticity.total()*varChromaticity.channels());
		double []  Temp = new double [sz]; 
		varChromaticity.get(0, 0,Temp);// Retrieve RG values for column(i)
		double [][] hist  = new double [histogramBin][histogramBin];// We're putting this in array list
		// Go thru all the RG values in col and fill array
		for(int val_RG = 0 ; val_RG < sz-1 ; val_RG = val_RG+2) {
			double val_R = Temp[val_RG] , val_G = Temp[val_RG+1];
			R = (int)Math.floor(val_R/bounds);
			G = (int)Math.floor(val_G/bounds);
			if(R == histogramBin) // check bounds
				R = histogramBin-1;
			if(G == histogramBin)
				G = histogramBin-1;
			hist[R][G]++;
		}
		double sum = 0;
		for(int i=0 ; i<histogramBin ; i++)
			for(int j=0 ; j<histogramBin ; j++)
				sum+=hist[i][j];
		for(int i=0 ; i<histogramBin ; i++)
			for(int j=0 ; j<histogramBin ; j++)
				hist[i][j]= hist[i][j]/sum;
		
		Histogram.add(hist);
	}
	protected void createFrameGrabber() throws InterruptedException { // modified frame grabber from A2 to support our function calls
		frameNumber = 0;
		if(capture != null && capture.isOpened()) { // The video must be opened
			double framePerSecond = capture.get(Videoio.CAP_PROP_FPS);
			frameCount = (int) Math.round(capture.get(Videoio.CAP_PROP_FRAME_COUNT));
			// Create a runnable to fetch new frames periodically
			Runnable frameGrabber = new Runnable() {
				@Override
				public void run() {
					Mat frame = new Mat();
					if (capture.read(frame)) {// Decode successfully
						Image im = Utilities.mat2Image(frame);
						Utilities.onFXThread(imageView.imageProperty(), im);
						if(frameNumber == 0) {
							image = frame;
						}
						createRowCol(null);
						chromaticity(null);
						image = frame;
						frameNumber++;
						}
					else 
						capture.release(); // Reach the end of the video
				}
			};

			for(int i = 0 ; i < frameCount ; i++)
				frameGrabber.run();
			 // terminate the timer if it is running
			if (timer != null && !timer.isShutdown()) {
				timer.shutdown();
				timer.awaitTermination(Math.round(1000/framePerSecond), TimeUnit.MILLISECONDS);
			}
			// run the frame grabber
			timer = Executors.newSingleThreadScheduledExecutor();
			timer.scheduleAtFixedRate(frameGrabber, 0, Math.round(1000/framePerSecond), TimeUnit.MILLISECONDS);
		}
	}

	// Hist STI calculator
	protected void histSTI(ActionEvent event){
		ArrayList<ArrayList<Double>> column_I = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> row_I = new ArrayList<ArrayList<Double>>();

		// Col STI
		histCol = new Mat (32,frameCount,CvType.CV_8UC3);
		for(int column = 0 ; column < 32; column++) {
			ArrayList<Mat> framecols = new ArrayList<>();	// Stores cols for each frame
			
			// Turn frame to frame # of cols
			for(int i = 0 ; i <frameCount ; i++) { 
				Mat newimg = new Mat();
				Imgproc.resize(frames.get(i), newimg, new Size(32,32));
				framecols.add(newimg.col(column));
			}
			// Create hist for each frame col
			ArrayList<double[][]> histogram = new ArrayList<>();
			for(int i = 0 ; i< frameCount ; i++) 
				createHistogram(histogram , framecols.get(i), framecols.get(i).rows());

			// Calculating variable I for each col
			ArrayList<Double> I = new ArrayList<>();
			histDiff(histogram , I);				
			column_I.add(I);
		}
		// Image fill
		for(int i = 0; i < column_I.size(); i++)
			for(int j = 0 ; j < column_I.get(0).size() ; j++) {
				double value = column_I.get(i).get(j) *255;
				double [] data = {value,value,value};
				histCol.put(i, j, data);
		}
	
		// Row STI
		histRow = new Mat (32,frameCount,CvType.CV_8UC3);
		for(int row = 0 ; row < 32; row++) {
			ArrayList<Mat> frameRow = new ArrayList<>();	// Stores cols for each frame
			
			// Turn frame to frame # of cols
			for(int i = 0 ; i <frameCount ; i++) {  
				Mat resizedImage = new Mat();
				Imgproc.resize(frames.get(i), resizedImage, new Size(32,32));
				frameRow.add(resizedImage.row(row));
			}
			
			// Create hist for each frame col
			ArrayList<double[][]> histogram = new ArrayList<>();
			for(int i = 0 ; i< frameCount ; i++)
				createHistogram(histogram , frameRow.get(i),frameRow.get(i).cols());
			

			// Calculate variable I for each col
			ArrayList<Double> I = new ArrayList<>();
			histDiff(histogram , I);
			row_I.add(I);// col_I will contain variable I for the image when the loop is done
			
		}
		// Image fill
		for(int i = 0; i < row_I.size(); i++)
			for(int j = 0 ; j < row_I.get(0).size() ; j++) {
				double val = row_I.get(i).get(j) *255;
				double [] items = {val,val,val};
				histRow.put(i, j, items);
			}
	}
}

