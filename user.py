import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image, ImageTk
import json
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

# Load the trained model
model = load_model("final_model_mobilenetv2.keras")

# Define class labels
class_labels = ['dhaner_shish', 'eagle', 'guri', 'langol', 'nouka']

# Initialize vote counts
vote_counts = {label: 0 for label in class_labels}

# JSON file for live updates
json_file = "live_votes.json"

def update_json():
    """Update the live_votes.json file with current vote counts."""
    with open(json_file, "w") as f:
        json.dump(vote_counts, f, indent=4)

def start_server():
    """Start an HTTP server to serve index.html and the JSON file."""
    handler = SimpleHTTPRequestHandler
    server = HTTPServer(("localhost", 8000), handler)
    print("Serving live voting at http://localhost:8000")
    server.serve_forever()

def classify_image(filepath):
    """Classify an image and return the predicted class."""
    try:
        # Preprocess the image
        img = load_img(filepath, target_size=(198, 400))  # Update target size
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify image: {e}")
        return None

def upload_image():
    """Handle image upload and classification."""
    filepath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if filepath:
        # Display the image
        img = Image.open(filepath)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

        # Classify the image
        result = classify_image(filepath)
        if result in vote_counts:
            vote_counts[result] += 1
            update_json()  # Update the JSON file
            result_label.config(text=f"Vote counted for: {result}")
        else:
            result_label.config(text="Error in prediction!")

def end_voting():
    """End voting and display the winner."""
    winner = max(vote_counts, key=vote_counts.get)
    winner_count = vote_counts[winner]
    
    # Save winner information to a JSON file
    winner_info = {
        "winner": winner,
        "count": winner_count,
        "logo": f"images/{winner}.png"  # Assuming logos are named according to the class labels
    }
    with open("winner.json", "w") as f:
        json.dump(winner_info, f, indent=4)

    # Create a custom popup window to show the winner
    winner_popup = tk.Toplevel(root)
    winner_popup.title("Voting Ended")
    winner_popup.geometry("300x200")
    winner_popup.configure(bg="lightblue")

    winner_label = tk.Label(winner_popup, text=f"The winner is: {winner}", font=("Arial", 16, "bold"), bg="lightblue", fg="darkgreen")
    winner_label.pack(pady=30)

    winner_count_label = tk.Label(winner_popup, text=f"Votes: {winner_count}", font=("Arial", 14), bg="lightblue", fg="darkgreen")
    winner_count_label.pack(pady=10)

    close_button = tk.Button(winner_popup, text="Close", command=winner_popup.destroy, bg="white", fg="black", font=("Arial", 12))
    close_button.pack(pady=10)

    winner_popup.mainloop()  # Keep the winner popup open

    root.quit()  # Close the main window after voting ends

def save_results(winner, winner_count):
    """Save voting results to a file."""
    try:
        with open("voting_results.txt", "w") as f:
            f.write("Voting Results:\n")
            for label, count in vote_counts.items():
                f.write(f"{label}: {count}\n")
            f.write(f"\nWinner: {winner} with {winner_count} votes!\n")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save results: {e}")

# Start the server in a separate thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Create the main application window
root = tk.Tk()
root.title("Smart Ballot")
root.geometry("400x600")
root.configure(bg="lightblue")

# Add GUI components
upload_button = tk.Button(root, text="Upload Ballot Image", command=upload_image, bg="white", fg="black", font=("Arial", 12, "bold"))
upload_button.pack(pady=10)

image_label = tk.Label(root, bg="lightblue")
image_label.pack(pady=10)

result_label = tk.Label(root, text="Scan: ", font=("Arial", 16), bg="lightblue", fg="darkgreen")
result_label.pack(pady=10)

end_button = tk.Button(root, text="End Voting and Show Winner", command=end_voting, bg="white", fg="black", font=("Arial", 12, "bold"))
end_button.pack(pady=10)

# Create an initial live_votes.json file
update_json()

# Run the application
root.mainloop()
