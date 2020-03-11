package application;

import java.io.*;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.regex.Pattern;
import java.util.concurrent.TimeUnit;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.FloatControl;
import javax.sound.sampled.Line;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.AudioInputStream;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;

import javafx.beans.InvalidationListener;
import javafx.beans.Observable;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;

import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.media.AudioClip;
import javafx.scene.media.MediaPlayer;
import javafx.scene.text.Text;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.media.Media;

import javafx.util.Duration;
import javafx.stage.FileChooser;
import utilities.Utilities;

public class Controller {
	
	@FXML
	private ImageView imageView; // the image display window in the GUI

	private Mat image;
	private int flag = 0;
	private boolean pl = true;
	private int width;
	private int height;
	private int sampleRate; // sampling frequency
	private int sampleSizeInBits;
	private int numberOfChannels;
	private double[] freq; // frequencies for each particular row
	private int numberOfQuantizionLevels;
	private int numberOfSamplesPerColumn;
	@FXML
	private Slider slider;
	@FXML
	private Slider volslider;
	@FXML 
	private Button pause;
	
	private VideoCapture capture;
	private ScheduledExecutorService timer;
	
	@FXML
	private void initialize() {
		// Optional: You should modify the logic so that the user can change these values
		// You may also do some experiments with different values
		width = 64;
		height = 64;
		sampleRate = 8000;
		sampleSizeInBits = 8;
		numberOfChannels = 1;
		
		numberOfQuantizionLevels = 16;
		
		numberOfSamplesPerColumn = 500;
		
		volslider.setMin(0);
		volslider.setMax(100);
		volslider.setShowTickLabels(true);
		volslider.setShowTickMarks(true);
		volslider.setMajorTickUnit(50);
		volslider.setMinorTickCount(5);
		volslider.setBlockIncrement(10);
		volslider.setValue(100);
		
		// assign frequencies for each particular row
		freq = new double[height]; // Be sure you understand why it is height rather than width
		freq[height/2-1] = 440.0; // 440KHz - Sound of A (La)
		for (int m = height/2; m < height; m++) {
			freq[m] = freq[m-1] * Math.pow(2, 1.0/12.0); 
		}
		for (int m = height/2-2; m >=0; m--) {
			freq[m] = freq[m+1] * Math.pow(2, -1.0/12.0); 
		}
		// rajan: volume controller
		volslider.setValue(100);
		volslider.valueProperty().addListener(new InvalidationListener() {
			@Override
			public void invalidated(Observable observable) {
				Mixer.Info [] mix = AudioSystem.getMixerInfo();  
				
				for (Mixer.Info mixerInfo : mix){
				    Mixer mixer = AudioSystem.getMixer(mixerInfo);
				    Line.Info[] info2 = mixer.getTargetLineInfo();
				    
				    for (Line.Info info:info2){
				        boolean open = true;  
				        Line line = null;  
				        
				        try {
				            line = mixer.getLine(info);  
				            open = line.isOpen() || line instanceof Clip;
				            if (!open)    
				                line.open();
				            FloatControl volCtrl = (FloatControl)line.getControl(FloatControl.Type.VOLUME);  
				            volCtrl.setValue((float)volslider.getValue()/100);
				        }  
				        
				        catch (LineUnavailableException e) {  
				            e.printStackTrace();  
				        }  
				        catch (IllegalArgumentException iaEx) {  
				        }  
				        
				        finally {  
				            if (line != null && !open) 
				                line.close();
				        }  
				    }
				}
			}

		});
	}
	// rajan: below function is commented out cuz we shouldnt need it, but i left it in just in case
//	private String getImageFilename() {
//		// This method should return the filename of the image to be played
//		// You should insert your code here to allow user to select the file
//		return "resources/test.mp4";
//	}
	@FXML
	protected void openImage(ActionEvent event) throws InterruptedException {
		// This method opens an image and display it using the GUI
		// You should modify the logic so that it opens and displays a video
		// Rajan: The following code block was provided by the Oracle Java Docs for JFileChooser with slight modification
	    JFileChooser chooser = new JFileChooser();
	    FileNameExtensionFilter filter = new FileNameExtensionFilter(
	        "jpg, gif, png, mp4, mov, wav, jpeg", "jpg", "gif", "png", "mp4", "mov", "wav", "jpeg");
	    chooser.setFileFilter(filter);
	    int returnVal = chooser.showOpenDialog(null);
	    if(returnVal == JFileChooser.APPROVE_OPTION) {
	       System.out.println("You chose to open this file: " +
	            chooser.getSelectedFile().getName());
	    }
	    
	    String file = chooser.getSelectedFile().getName(); // rajan: name of video
	    String filetype = file.substring(file.lastIndexOf("."),file.length()); // rajan: name of extension (.jpg, .mp4, etc.)

	    // Rajan: ALL SELECTED FILES MUST BE IN RESOURCES FOLDER TO WORK!!!
	    // rajan : below code decides whether picture is image or video. issue to fix later: are .gif files images or videos?
	    if(filetype.equals(".mp4") || filetype.equals(".mov") || filetype.equals(".wav") || filetype.equals(".gif")) {
			capture = new VideoCapture("resources/" + file); // open video file
			if (capture.isOpened()) { // open successfully
				createFrameGrabber();
			} 
	    }
	    if(filetype.equals(".png") || filetype.equals(".jpg") || filetype.equals(".jpeg")) {
			final String imageFilename = "resources/" + file;
			image = Imgcodecs.imread(imageFilename);
			imageView.setImage(Utilities.mat2Image(image)); 
	    }
		// You don't have to understand how mat2Image() works. 
		// In short, it converts the image from the Mat format to the Image format
		// The Mat format is used by the opencv library, and the Image format is used by JavaFX
		// BTW, you should be able to explain briefly what opencv and JavaFX are after finishing this assignment
	}
	private void playAudio() { // rajan: this plays the click noise
		File sound = new File("resources/click.wav");
		try {
			Clip click = AudioSystem.getClip();
			click.open(AudioSystem.getAudioInputStream(sound));
			click.start();
				
			Thread.sleep(click.getMicrosecondLength()/1000);
		}
		catch(Exception e) {
				
		}
	}
	protected void createFrameGrabber() throws InterruptedException {
		 if (capture != null && capture.isOpened()) { // the video must be open
			 System.out.println("Video open"); // Rajan: Check if the video has been opened
			 double framePerSecond = capture.get(Videoio.CAP_PROP_FPS);
			 // create a runnable to fetch new frames periodically
			 Runnable frameGrabber = new Runnable() {
				 @Override
				 public void run() {
					 Mat frame = new Mat();
					 if (capture.read(frame)) { // decode successfully
						 if(pl) { // check if video is paused
							 Image im = Utilities.mat2Image(frame);
							 Utilities.onFXThread(imageView.imageProperty(), im);
							 image = frame;
							 double currentFrameNumber = capture.get(Videoio.CAP_PROP_POS_FRAMES); // current frame number
							 double totalFrameCount = capture.get(Videoio.CAP_PROP_FRAME_COUNT);   // total frame count
							 slider.setValue(currentFrameNumber / totalFrameCount * (slider.getMax() - slider.getMin())); //this sets slider position
							 if(currentFrameNumber%2==0){
								 if (flag!=1)
								 playAudio();
							 }
						 }
						 else {
							 while(!pl) { //if paused, send into sleep loop
								 try {
									Thread.sleep(1);
								} catch (InterruptedException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}
							 }
						 }
					 } 
					 else { // reach the end of the video
						 capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
					 }
				 }
			 };
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
	@FXML
	protected void pause(ActionEvent event) throws InterruptedException{ // pause/play feature
		if(pl) {
		pause.setText("Play");
		}
		if(!pl) {
			pause.setText("Pause");
		}
		pl = !pl;
	}

	@FXML
	protected void playImage(ActionEvent event) throws LineUnavailableException {
		// This method "plays" the image opened by the user
		// You should modify the logic so that it plays a video rather than an image
		if (image != null) {
			flag = 1;			
			// convert the image from RGB to grayscale
			Mat grayImage = new Mat();
			Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);
			
			// resize the image
			Mat resizedImage = new Mat();
			Imgproc.resize(grayImage, resizedImage, new Size(width, height));
			
			// quantization
			double[][] roundedImage = new double[resizedImage.rows()][resizedImage.cols()];
			for (int row = 0; row < resizedImage.rows(); row++) {
				for (int col = 0; col < resizedImage.cols(); col++) {
					roundedImage[row][col] = (double)Math.floor(resizedImage.get(row, col)[0]/numberOfQuantizionLevels) / numberOfQuantizionLevels;
				}
			}
			
			// I used an AudioFormat object and a SourceDataLine object to perform audio output. Feel free to try other options
	        AudioFormat audioFormat = new AudioFormat(sampleRate, sampleSizeInBits, numberOfChannels, true, true);
            SourceDataLine sourceDataLine = AudioSystem.getSourceDataLine(audioFormat);
            sourceDataLine.open(audioFormat, sampleRate);
            sourceDataLine.start();
            for (int col = 0; col < width; col++) {
            	byte[] audioBuffer = new byte[numberOfSamplesPerColumn];
            	for (int t = 1; t <= numberOfSamplesPerColumn; t++) {
            		double signal = 0;
                	for (int row = 0; row < height; row++) {
                		int m = height - row - 1; // Be sure you understand why it is height rather width, and why we subtract 1 
                		int time = t + col * numberOfSamplesPerColumn;
                		double ss = Math.sin(2 * Math.PI * freq[m] * (double)time/sampleRate);
                		signal += roundedImage[row][col] * ss;
                	}
                	double normalizedSignal = signal / height; // signal: [-height, height];  normalizedSignal: [-1, 1]
                	audioBuffer[t-1] = (byte) (normalizedSignal*0x7F); // Be sure you understand what the weird number 0x7F is for
            	}
            	sourceDataLine.write(audioBuffer, 0, numberOfSamplesPerColumn);
            }
            sourceDataLine.drain();
            sourceDataLine.close();
            flag = 0;
		} else {
			System.out.println("No selected image.");
		}
	} 
}
