import cv2
import easyocr
import os
import time
import numpy as np

def extract_text_from_frame(frame, reader, min_confidence=0.5):  # Added min_confidence parameter
    """Extracts text from a single frame using EasyOCR.

    Args:
        frame (numpy.ndarray): The image frame to process.
        reader (easyocr.Reader): The EasyOCR reader instance.
        min_confidence (float): Minimum confidence score to keep text (default: 0.5).

    Returns:
        list: A list of dictionaries containing the extracted text, coordinates, and confidence score.
    """
    results = reader.readtext(frame)
    extracted_text = []
    for (bbox, text, prob) in results:
        if prob >= min_confidence:  # Check against min_confidence
            extracted_text.append({
                "text": text,
                "coordinates": [(int(x), int(y)) for (x, y) in bbox],
                "confidence": prob
            })
    return extracted_text

def is_frame_unclear(frame, threshold=70):
    if frame is None:
        return True # Consider the frame unclear if it's None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value < threshold

def process_video(video_path, frame_interval=2, languages=['en', 'no'], output_dir='extracted_frames', min_text_confidence=0.5):
    """Processes a video, extracts text, and saves frames as images in their original size.

    Args:
        video_path (str): The path to the video file.
        frame_interval (int): The number of frames to skip between extractions.
        languages (list): A list of languages to use for OCR (default: ['en', 'no']).
        output_dir (str): The directory to save the extracted frames.
        min_text_confidence (float): Minimum confidence score to keep extracted text (default: 0.5).
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = round(fps / frame_interval)

    reader = easyocr.Reader(languages)

    frame_count = 0
    extracted_data = []

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video, exit loop

        if frame_count % interval == 0 or frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1:

            while is_frame_unclear(frame) and ret:
                ret, frame = cap.read()
                if not ret:
                    break

            # Check again if a frame was successfully read after the loop
            if ret:
                extracted_data.extend(extract_text_from_frame(frame, reader, min_text_confidence))
                output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()
    return extracted_data


if __name__ == "__main__":

    start_time = time.time()
    print("Start Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    video_path = 'kjoleskap.mp4'  # Replace with your video path
    extracted_data = process_video(video_path, languages=['no'], min_text_confidence=0.7)



    for item in extracted_data:
        print("Text:", item["text"])
        print("Coordinates:", item["coordinates"])
        print("---")

    end_time = time.time()
    print("End Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))

    # Calculate Duration
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    print(f"Total Processing Time: {minutes} minutes, {seconds} seconds")