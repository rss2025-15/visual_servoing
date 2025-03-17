import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
    """
    Helper function to print out images for debugging.
    Press any key to continue.
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cd_color_segmentation(img, template=None, linefollower=False):
    """
    Implement color segmentation to detect a cone (or any object).

    Inputs:
        img (np.ndarray): The input BGR image in which to detect the cone.
        template (str or None): Optional file path to a template image
                                for automatically determining HSV filter ranges.
                                If None, we'll use fixed HSV bounds.

    Returns:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the detected object in px
              (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
    """
    # 1) Convert the main image to HSV
    if linefollower:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = 190
        hsv_img = hsv_img[lower_bound:lower_bound+60, :]
    else:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # -------------------------------------------------
    # 2) If a template path is given, derive HSV filter
    #    ranges from that template automatically.
    #    Otherwise, we use a fixed example range.
    # -------------------------------------------------

    if template is not None:
        # Read the template image from file
        template_bgr = cv2.imread(template)
        if template_bgr is None:
            raise FileNotFoundError(f"Could not read template image: {template}")

        template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)

        # (OPTIONAL) If your template contains background, crop to just the object ROI:
        # Adjust these bounding box coordinates as needed
        # e.g. roi = template_hsv[ymin:ymax, xmin:xmax]
        roi = template_hsv  # If the template is already mostly the object

        # Compute min & max per channel in the ROI
        H, S, V = roi[:,:,0], roi[:,:,1], roi[:,:,2]


        # Define thresholds to remove white pixels
        white_threshold_v = 200  # V > 200 is considered white
        white_threshold_s = 50   # S < 50 means low saturation (nearly white)

        # Create a mask for non-white pixels
        non_white_mask = (V < white_threshold_v) & (S > white_threshold_s)

        # Apply the mask to filter out white pixels
        H = H[non_white_mask]
        S = S[non_white_mask]
        V = V[non_white_mask]
        minH, maxH = np.min(H), np.max(H)
        print(minH, maxH)


        # OpenCV H range = [0..179], S & V range = [0..255]
        lower_hsv = np.array([
            max(minH , 0),
            100,
            100,
        ], dtype=np.uint8)
        upper_hsv = np.array([
            maxH,
            255,
            255
        ], dtype=np.uint8)

    else:
        # If no template is supplied, just use a fixed “orange-ish” example range
        # (Adjust these to your object’s color as needed)
        lower_hsv = np.array([2, 100, 100], dtype=np.uint8)
        upper_hsv = np.array([15, 255, 255], dtype=np.uint8)

    # -------------------------------------------------
    # 3) Threshold the main image (hsv_img) using our HSV bounds
    # -------------------------------------------------
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    # combined_view = np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
    # image_print(combined_view)
    # -------------------------------------------------
    # 4) (OPTIONAL) Morphological operations to reduce noise
    # -------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=0)
    # mask = cv2.dilate(mask, kernel, iterations=2)


    # -------------------------------------------------
    # 5) Find contours on the mask; pick the largest one
    # -------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_box = ((0, 0), (0, 0))  # Default if no object found
    largest_area = 0
    best_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            best_contour = cnt

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        if linefollower:
            bounding_box = ((x, y + lower_bound), (x + w, y + h + lower_bound))
        else:
            bounding_box = ((x, y), (x + w, y + h))
    return bounding_box

def cd_color_segmentation_line(img, template=None):
    """
    Implement color segmentation to detect a cone (or any object).

    Inputs:
        img (np.ndarray): The input BGR image in which to detect the cone.
        template (str or None): Optional file path to a template image
                                for automatically determining HSV filter ranges.
                                If None, we'll use fixed HSV bounds.

    Returns:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the detected object in px
              (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
    """
    # 1) Convert the main image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = 120
    hsv_img = hsv_img[lower_bound:lower_bound+120, :]

    # -------------------------------------------------
    # 2) If a template path is given, derive HSV filter
    #    ranges from that template automatically.
    #    Otherwise, we use a fixed example range.
    # -------------------------------------------------

    if template is not None:
        # Read the template image from file
        template_bgr = cv2.imread(template)
        if template_bgr is None:
            raise FileNotFoundError(f"Could not read template image: {template}")

        template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)

        # (OPTIONAL) If your template contains background, crop to just the object ROI:
        # Adjust these bounding box coordinates as needed
        # e.g. roi = template_hsv[ymin:ymax, xmin:xmax]
        roi = template_hsv  # If the template is already mostly the object

        # Compute min & max per channel in the ROI
        H, S, V = roi[:,:,0], roi[:,:,1], roi[:,:,2]


        # Define thresholds to remove white pixels
        white_threshold_v = 200  # V > 200 is considered white
        white_threshold_s = 50   # S < 50 means low saturation (nearly white)

        # Create a mask for non-white pixels
        non_white_mask = (V < white_threshold_v) & (S > white_threshold_s)

        # Apply the mask to filter out white pixels
        H = H[non_white_mask]
        S = S[non_white_mask]
        V = V[non_white_mask]
        minH, maxH = np.min(H), np.max(H)
        print(minH, maxH)


        # OpenCV H range = [0..179], S & V range = [0..255]
        lower_hsv = np.array([
            max(minH , 0),
            100,
            100,
        ], dtype=np.uint8)
        upper_hsv = np.array([
            maxH,
            255,
            255
        ], dtype=np.uint8)

    else:
        # If no template is supplied, just use a fixed “orange-ish” example range
        # (Adjust these to your object’s color as needed)
        lower_hsv = np.array([5, 100, 50], dtype=np.uint8)
        upper_hsv = np.array([15, 255, 150], dtype=np.uint8)

    # -------------------------------------------------
    # 3) Threshold the main image (hsv_img) using our HSV bounds
    # -------------------------------------------------
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    # -------------------------------------------------
    # 4) (OPTIONAL) Morphological operations to reduce noise
    # -------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # -------------------------------------------------
    # 5) Find contours on the mask; pick the largest one
    # -------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_box = ((0, 0), (0, 0))  # Default if no object found
    largest_area = 0
    best_contour = None
    # img dimension is 360, 640

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            best_contour = cnt

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        bounding_box = ((x, y + lower_bound), (x + w, y + h + lower_bound))
    return bounding_box
# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
# Load the image
    FILE_NAME = "test_images_cone/test19.jpg"  # Update this if needed
    template = "test_images_cone/cone_template.png"
    # FILE_NAME = "test_images_citgo/citgo10.jpeg"  # Update this if needed
    # template = "test_images_citgo/citgo_template.png"
    # 1) Load a test image (where you want to detect the cone or logo)
    test_image_path = "test_scene.jpg"
    test_img = cv2.imread(FILE_NAME)
    # image_print(test_img)
    # 2) Call the detection function
    #    Option A: use a template file to auto-derive HSV bounds

    bbox = cd_color_segmentation(test_img, template)

    #    Option B: omit the template to use fixed HSV bounds
    bbox = cd_color_segmentation(test_img)

    print("Bounding Box:", bbox)




    # 3) Visualize by drawing the bounding box on the test image
    (x1, y1), (x2, y2) = bbox
    cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)






    # 4) Display the result
    image_print(test_img)
