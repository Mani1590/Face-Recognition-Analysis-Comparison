import cv2
import os

# Write a program to input a img and apply gaussian blur on it
# Read the image
image_path = input("Enter the path to the image: ")
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # Create the directory if it doesn't exist
    output_dir = "Blurred imgs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the original image name and create the new image name
    original_image_name = os.path.basename(image_path)
    blurred_image_name = f"Blurred_{original_image_name}"
    blurred_image_path = os.path.join(output_dir, blurred_image_name)

    # Save the blurred image
    cv2.imwrite(blurred_image_path, blurred_image)

    # Display the original and blurred images
    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred_image)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
