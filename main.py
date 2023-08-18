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

    if input_source.isdigit():
        process_video(sess, anchors, input_source, classes)
    else:
        process_image(sess, anchors, input_source, classes)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_source = input(
        "Enter the path to the image or video file, or a camera device number: ")
    model_path = "./model/yolov4.onnx"
    anchors_path = "./model_data/anchors.txt"
    class_file_path = "./model_data/coco.names"

    main(input_source, model_path, anchors_path, class_file_path)
