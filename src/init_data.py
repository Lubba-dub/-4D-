import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Try to import scikit-image data
try:
    from skimage import data, img_as_ubyte
    from skimage.transform import resize
    print("scikit-image found.")
except ImportError:
    print("scikit-image not found. Please install it.")
    exit(1)

DATA_DIR = os.path.join("DIP_Project", "data")

def save_image(img, name):
    # Normalize to 0-255 and convert to uint8
    img_uint8 = img_as_ubyte(img)
    path = os.path.join(DATA_DIR, name)
    cv2.imwrite(path, img_uint8)
    print(f"Saved {name} to {path}")

def main():
    # 1. Shepp-Logan Phantom (Simulated Head CT)
    phantom = data.shepp_logan_phantom()
    phantom = resize(phantom, (512, 512), anti_aliasing=True)
    save_image(phantom, "phantom.png")

    # 2. Human Brain MRI (if available, else use a generated one)
    try:
        brain = data.brain()
        # brain is usually a 3D stack, take the 10th slice
        brain_slice = brain[9] 
        save_image(brain_slice, "brain_mri.png")
    except Exception as e:
        print(f"Could not load specific brain data: {e}")
        # Fallback: Create a noisy phantom to simulate complex structure
        noisy_phantom = phantom + 0.1 * np.random.randn(*phantom.shape)
        noisy_phantom = np.clip(noisy_phantom, 0, 1)
        save_image(noisy_phantom, "noisy_phantom.png")

if __name__ == "__main__":
    main()
