
import cv2
import numpy as np

def insert_advertisement(video_path, ad_image_path, output_path=None):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Load advertisement image
    ad_img = cv2.imread(ad_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if exists

    # Define the codec and create VideoWriter object
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    # Resize advertisement image to desired size
    ad_img_resized = cv2.resize(ad_img, (150, 150))  # Adjust size as needed

    # Convert advertisement image to grayscale
    ad_gray = cv2.cvtColor(ad_img_resized, cv2.COLOR_BGR2GRAY)

    # Threshold the advertisement image to create a mask for the background
    _, mask = cv2.threshold(ad_gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize video frame to desired size
        frame_resized = cv2.resize(frame, (500, 400))  # Adjust size as needed

        # Apply the mask to the advertisement image to remove the background
        ad_img_masked = cv2.bitwise_and(ad_img_resized, ad_img_resized, mask=mask_inv)

        # Define position to overlay the advertisement image
        x_offset = 10
        y_offset = 10

        # Occlusion Handling
        # Blend the advertisement image with the video frame
        frame_bg = cv2.bitwise_and(frame_resized[y_offset:y_offset+ad_img_resized.shape[0],
                                      x_offset:x_offset+ad_img_resized.shape[1]],
                                    frame_resized[y_offset:y_offset+ad_img_resized.shape[0],
                                                  x_offset:x_offset+ad_img_resized.shape[1]],
                                    mask=mask)
        frame_fg = cv2.bitwise_and(ad_img_masked, ad_img_masked, mask=mask_inv)
        frame_resized[y_offset:y_offset+ad_img_resized.shape[0],
                      x_offset:x_offset+ad_img_resized.shape[1]] = cv2.add(frame_bg, frame_fg)

        # Display the frame
        cv2.imshow('Video with Advertisement', frame_resized)

        # Write the frame to the output video if an output path is provided
        if output_path is not None:
            out.write(frame_resized)

        # Check for key press to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    if output_path is not None:
        out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify paths to input video, advertisement image, and output video
    video_path = 'E:\AIML Assign\AIML Task\Input Video 2.mp4'
    ad_image_path = 'E:\AIML Assign\AIML Task\Advertisement Image.jpg'
    output_path = 'E:\AIML Assign\AIML Task\Sample Video.mp4'

    # Call the function to insert advertisement into the video
    insert_advertisement(video_path, ad_image_path, output_path)
