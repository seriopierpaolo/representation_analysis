import cv2

def pad_resize(image, target_size=(128,128)):
    height, width = image.shape[:2]
    target_height, target_width = target_size

    # Calculate padding to make the image with the same width and height, like a square
    pad_dim = max(width, height)
    
    # Calculate total padding needed
    total_pad_width = pad_dim - width
    total_pad_height = pad_dim - height

    # Calculate top/bottom and left/right padding for centering
    pad_top = total_pad_height // 2
    pad_bottom = total_pad_height - pad_top
    pad_left = total_pad_width // 2
    pad_right = total_pad_width - pad_left

    # Apply padding
    padded_image = cv2.copyMakeBorder(
        image, 
        pad_top, 
        pad_bottom, 
        pad_left, 
        pad_right, 
        cv2.BORDER_CONSTANT, 
        value=0
    )

    # Resize to target size
    # Note: Using target_size=(target_width, target_height) for cv2.resize as expected
    resized_image = cv2.resize(padded_image, (target_width, target_height))

    return resized_image