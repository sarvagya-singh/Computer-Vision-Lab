import cv2


def main():
    
    video_path = '/home/Student/Downloads/4711694-uhd_3840_2160_30fps.mp4'

  
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 800, 600)

   
    while True:
        ret, frame = cap.read()

        if not ret:
            break

  
        cv2.imshow('Video', frame)

   
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
