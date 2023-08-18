import cv2
import onnxruntime as rt

from utils.detection import get_anchors, process_video, process_image


def main(input_source: str, model_path: str, anchors_path: str, class_file_path: str) -> None:
    """
    Main function to process object detection on given input source.

    Args:
        input_source (str): Path to the video or image file, or device index.
        model_path (str): Path to the ONNX model file.
        anchors_path (str): Path to the anchors file.
        class_file_path (str): Path to the file containing class names.

    Returns:
        None
    """

    # Create session options
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    sess_options.enable_profiling = False

    # Create session with the specified options
    sess = rt.InferenceSession(model_path, sess_options)

    # Check if GPU is available, and set CUDA as provider if so
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        sess.set_providers(['CUDAExecutionProvider'])

    anchors = get_anchors(anchors_path)

    classes = {}
    with open(class_file_path, 'r') as data:
        for ID, name in enumerate(data):
            classes[ID] = name.strip('\n')

    try:
        camera_index = int(input_source)
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            process_video(sess, anchors, camera_index, classes)
            cap.release()
            return
    except ValueError:
        pass

    # Try to read as an image
    image_test = cv2.imread(input_source)
    if image_test is not None:
        process_image(sess, anchors, input_source, classes)
        return

    # Try to read as a video
    cap = cv2.VideoCapture(input_source)
    if cap.isOpened():
        process_video(sess, anchors, input_source, classes)
        cap.release()
        return

    print(
        f"Failed to open the source: {input_source}. "
        "Please provide a valid image, video, or camera index."
    )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_source = input(
        "Enter the path to the image or video file, or a camera device number: ")
    model_path = "./model/yolov4.onnx"
    anchors_path = "./model_data/anchors.txt"
    class_file_path = "./model_data/coco.names"

    main(input_source, model_path, anchors_path, class_file_path)
