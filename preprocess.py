import cv2
import os
def crop_image(image_path, top_left, bottom_right):
    # Load the image
    image = cv2.imread(image_path)

    # Extract coordinates for cropping
    x1, y1 = top_left  # Top-left corner (x, y)
    x2, y2 = bottom_right  # Bottom-right corner (x, y)

    # Crop the image using slicing
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image
def cropping(display_rows,display_columns,crop_height,crop_width,binary,filename_map,output_dir):
    for i, r in enumerate(display_rows):
        for j, c in enumerate(display_columns):
            # Calculate coordinates for cropping
            start_y = r * crop_height
            end_y = (r + 1) * crop_height
            start_x = c * crop_width
            end_x = (c + 1) * crop_width

            # Crop the image using numpy slicing
            cropped_region = binary[start_y:end_y, start_x:end_x]

            resized_region = cv2.resize(cropped_region, (28, 28), interpolation=cv2.INTER_AREA)

            if (r, c) in filename_map:
                filename = filename_map[(r, c)] + '.png'
                output_path = os.path.join(output_dir, filename)

                # Save the resized image to the output directory
                cv2.imwrite(output_path, resized_region)
