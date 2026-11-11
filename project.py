import cv2
import numpy as np
import ttkbootstrap as tb
from tkinter import filedialog, Label, Frame, StringVar, Scale
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class PhotoNoiseReductionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Noise Reduction")
        self.root.geometry("1100x700")
        self.root.configure(bg="#D0E6F2")  # Soft light blue background
        self.root.resizable(True, True)

        self.img = None
        self.denoised_img = None
        self.original_img = None
        self.status_text = StringVar(value="Select an image to begin")

        self.create_widgets()

    def create_widgets(self):
        """Create and arrange all the GUI components."""
        # Create main frame
        main_frame = Frame(self.root, bg="#B2C9E6")  # Light blue background for main frame
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Create left and right frames
        self.left_frame = Frame(main_frame, bg="#FFDAB9", relief="ridge", bd=5)  # Peach color
        self.left_frame.pack(side="left", fill="y", padx=15, pady=15)

        self.right_frame = Frame(main_frame, bg="#87CEFA", relief="ridge", bd=5)  # Sky blue color
        self.right_frame.pack(side="right", expand=True, fill="both", padx=15, pady=15)

        # Status label
        status_label = tb.Label(self.root, textvariable=self.status_text, font=("Arial", 13, "italic"),
                                bootstyle="light", background="#D0E6F2", foreground="black")
        status_label.pack(pady=10)

        # Noise reduction rate variable
        self.noise_reduction_rate = Scale(self.left_frame, from_=1, to=100, orient="horizontal",
                                          label="Noise Reduction Rate (%)",
                                          command=lambda x: self.apply_noise_reduction(),
                                          bg="#FFDAB9", sliderlength=30)
        self.noise_reduction_rate.set(50)  # Default to 50%
        self.noise_reduction_rate.pack(pady=10, fill="x")

        # Create buttons
        self.create_button("📷 Select Image", self.select_image, "warning")
        self.create_button("💾 Save Image", self.save_image, "primary")
        self.create_button("📊 View Histogram", self.view_noise_reduction, "info")
        self.create_button("🔄 Reset to Original Image", self.reset_to_original, "light")

        # Panel for displaying images
        self.panel = None

    def create_button(self, text, command, color):
        """Create and place a button in the left frame."""
        btn = tb.Button(self.left_frame, text=text, command=command, bootstyle=color, padding=12)
        btn.pack(pady=10, ipadx=20, ipady=7, fill="x")

    def select_image(self):
        """Select an image file and load it."""
        img_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"),
                                                         ("JPEG files", "*.jpg"),
                                                         ("All Files", "*.*")])
        if img_path:
            self.original_img = cv2.imread(img_path)
            self.img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            self.status_text.set("Image loaded! Applying noise reduction...")
            self.apply_noise_reduction()

    def apply_noise_reduction(self):
        """Apply noise reduction to the image based on the selected rate."""
        if self.img is None:
            self.status_text.set("No image selected!")
            return

        # Get the noise reduction rate (from 1 to 100)
        rate = self.noise_reduction_rate.get()
        strength = rate / 10  # Adjusting the strength based on the percentage

        self.denoised_img = cv2.fastNlMeansDenoisingColored(self.img, None, strength, strength, 7, 21)
        self.status_text.set(f"Noise Reduction Rate {rate}% Applied!")
        self.display_image(self.denoised_img)

    def display_image(self, image):
        """Display the given image in the right frame."""
        image = Image.fromarray(image)
        image.thumbnail((600, 600), Image.LANCZOS)
        image = ImageTk.PhotoImage(image)

        if self.panel is None:
            self.panel = Label(self.right_frame, image=image, bg="#87CEFA")
            self.panel.image = image
            self.panel.pack(padx=10, pady=10, expand=True)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def save_image(self):
        """Save the processed image to a file."""
        if self.denoised_img is None:
            self.status_text.set("No processed image to save!")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All Files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(self.denoised_img, cv2.COLOR_RGB2BGR))
            self.status_text.set("Image saved successfully!")

    def view_noise_reduction(self):
        """Display histograms comparing original and denoised images."""
        if self.img is None or self.denoised_img is None:
            self.status_text.set("Apply noise reduction first!")
            return

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        denoised_gray = cv2.cvtColor(self.denoised_img, cv2.COLOR_RGB2GRAY)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(img_gray.ravel(), bins=256, color='#FFBA08', alpha=0.8, label="Original")
        plt.title("Original Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(denoised_gray.ravel(), bins=256, color='#FFDD44', alpha=0.8, label="Denoised")
        plt.title("Denoised Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()

        plt.suptitle("Noise Reduction Effect")
        plt.tight_layout()
        plt.show()

    def reset_to_original(self):
        """Reset the image to the original and clear any noise reduction."""
        if self.original_img is None:
            self.status_text.set("No original image to reset!")
            return
        self.img = self.original_img
        self.denoised_img = None
        self.display_image(self.img)
        self.status_text.set("Reset to original image!")


if __name__ == "__main__":
    root = tb.Window(themename="superhero")
    app = PhotoNoiseReductionApp(root)
    root.mainloop()
ss