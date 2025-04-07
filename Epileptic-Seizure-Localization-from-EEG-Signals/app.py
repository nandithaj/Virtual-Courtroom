# app.py

import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import font as tkfont
import os
import csv
import threading
from PIL import Image, ImageTk, ImageOps
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from seizure_prediction import PredictSeizure
from shap_explainer import ShapExplainer
from lime_explainer import LimeExplainer

# Function to check credentials from credentials.csv
def check_credentials(username, password):
    if not os.path.exists("credentials.csv"):
        messagebox.showerror("Error", "credentials.csv file not found!")
        return False

    with open("credentials.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["USERNAME"] == username and row["PASSWORD"] == password:
                return True
    return False

# Function to handle login
def login():
    username = username_entry.get()
    password = password_entry.get()

    if check_credentials(username, password):
        messagebox.showinfo("Login Successful", "Welcome!")
        login_screen.destroy()
        open_main_app(username)
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Function to open the main application
def open_main_app(username):
    main_app = tk.Tk()
    main_app.title("Main Application")
    main_app.state('zoomed')

    # Top bar frame with 70px height
    top_bar = tk.Frame(main_app, bg="#2176BD", height=70)
    top_bar.pack(fill=tk.X, side=tk.TOP)
    top_bar.pack_propagate(False)  # Prevent frame from resizing based on contents

    # Display name with same style as login page
    name = None
    with open("credentials.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['USERNAME'] == username:
                name = row['NAME']
                break

    # Welcome label with adjusted styling for 70px height
    welcome_font = tkfont.Font(size=20, weight='bold')  # Reduced from 22 to 20
    username_label = tk.Label(top_bar, 
                            text=f"Welcome, Dr. {name}", 
                            bg="#2176BD",  # Changed to match top_bar color
                            fg="white",
                            font=welcome_font)
    username_label.pack(side=tk.LEFT, padx=20, pady=10)  # Reduced vertical padding

    # Improved Exit button with adjusted size for 70px height
    logout_button = tk.Button(top_bar, 
                            text="Exit", 
                            bg="#FFFFFF",
                            fg="#2176BD",  # Changed to match top_bar color
                            font=("Arial", 10, "bold"),  # Reduced from 11 to 10
                            activebackground="#F0F0F0",
                            activeforeground="#2176BD",
                            relief=tk.FLAT,
                            bd=0,
                            padx=15,  # Reduced from 18
                            pady=5,   # Reduced from 8
                            command=main_app.destroy)
    logout_button.pack(side=tk.RIGHT, padx=20, pady=10)  # Reduced vertical padding

    # Hover effects
    def on_enter(e):
        logout_button['background'] = '#F0F0F0'
        logout_button['foreground'] = '#2176BD'

    def on_leave(e):
        logout_button['background'] = '#FFFFFF'
        logout_button['foreground'] = '#2176BD'

    logout_button.bind("<Enter>", on_enter)
    logout_button.bind("<Leave>", on_leave)

    # Select patient function
    def select_patient():
        global patient_path
        patient_path = filedialog.askdirectory()
        if patient_path:
            profile_path = os.path.join(patient_path, "profile.txt")
            with open(profile_path, "r") as profile_file:
                profile_data = profile_file.readlines()                
                notes = "".join(profile_data[2:]) 

                # Clear existing entries
                name_entry.delete(0, "end")
                age_entry.delete(0, "end")
                notes_entry.delete("1.0", tk.END)
                
                # Insert new data
                name_entry.insert(0, profile_data[0].strip())
                age_entry.insert(0, profile_data[1].strip())
                notes_entry.insert("1.0", notes)

    # Update patient details function
    def update_patient_details():
        global patient_path
        if not patient_path:
            messagebox.showwarning("No Patient", "Please select a patient first")
            return
        
        try:
            # Validate required fields
            if not name_entry.get().strip():
                messagebox.showwarning("Validation", "Name cannot be empty")
                return
            if not age_entry.get().strip():
                messagebox.showwarning("Validation", "Age cannot be empty")
                return
                
            new_data = [
                name_entry.get().strip(),
                age_entry.get().strip(),
                notes_entry.get("1.0", tk.END).strip()
            ]
            
            profile_path = os.path.join(patient_path, "profile.txt")
            
            # Create directory if it doesn't exist
            os.makedirs(patient_path, exist_ok=True)
            
            with open(profile_path, "w") as profile_file:
                profile_file.write("\n".join(new_data))
            
            messagebox.showinfo("Success", "Profile saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profile")
    
    def view_edf_data():
        edf_data_path = os.path.join(patient_path, 'edf_data')
        if not os.path.exists(edf_data_path):
            messagebox.showerror("Error", "EDF data folder not found")
            return

        files = [f for f in os.listdir(edf_data_path) if f.endswith(".edf")]
        if not files:
            messagebox.showinfo("Error", "No EDF files found in the folder")
            return

        # Main file selection popup
        popup = tk.Toplevel(main_app)
        popup.title("Select EDF File")
        popup.geometry("400x300")
        
        # Top bar
        top_bar = tk.Frame(popup, bg="#2176BD", height=50)
        top_bar.pack(fill="x", side="top")
        tk.Label(top_bar, text="Select a file to display:", bg="#2176BD", 
                fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        # Listbox with scrollbar
        list_frame = tk.Frame(popup)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        listbox.pack(fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        for file in sorted(files):
            listbox.insert(tk.END, file)

        # Channel selection function
        def select_channels(channels):
            selected = []
            channel_popup = tk.Toplevel(popup)
            channel_popup.title("Select Channels")
            channel_popup.geometry("500x450")
            
            # Top bar
            top_bar = tk.Frame(channel_popup, bg="#2176BD", height=50)
            top_bar.pack(fill="x")
            tk.Label(top_bar, text="Select channels to display:", 
                    bg="#2176BD", fg="white", font=("Arial", 11, "bold")).pack(pady=10)
            
            # Canvas with scrollable frame for checkboxes
            canvas = tk.Canvas(channel_popup)
            scrollbar = tk.Scrollbar(channel_popup, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
            
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="top", fill="both", expand=True)
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            def on_frame_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
                
            scrollable_frame.bind("<Configure>", on_frame_configure)
            
            # Checkboxes in grid layout (all unchecked by default)
            checkboxes = []
            max_columns = 5
            
            for i, channel in enumerate(sorted(channels)):
                var = tk.StringVar(value="")
                cb = tk.Checkbutton(
                    scrollable_frame,
                    text=channel,
                    variable=var,
                    onvalue=channel,
                    offvalue="",
                )
                row, col = divmod(i, max_columns)
                cb.grid(row=row, column=col, sticky="w", padx=5, pady=2)
                checkboxes.append((cb, var, channel))  # Now storing channel name too
            
            # Button frame (below checkboxes)
            button_frame = tk.Frame(channel_popup)
            button_frame.pack(side="top", fill="x", pady=10)
            
            # Fixed Select All/Deselect All functionality
            def select_all():
                for cb, var, channel in checkboxes:
                    var.set(channel)  # Set to channel name (onvalue)
                    cb.select()  # Visually select the checkbox
            
            def deselect_all():
                for cb, var, channel in checkboxes:
                    var.set("")  # Set to empty string (offvalue)
                    cb.deselect()  # Visually deselect the checkbox
            
            # Selection buttons in one line
            select_buttons_frame = tk.Frame(button_frame)
            select_buttons_frame.pack(side="left", padx=10)
            
            tk.Button(select_buttons_frame, text="Select All", command=select_all).pack(side="left", padx=5)
            tk.Button(select_buttons_frame, text="Deselect All", command=deselect_all).pack(side="left", padx=5)
            
            # OK/Cancel buttons
            ok_cancel_frame = tk.Frame(button_frame)
            ok_cancel_frame.pack(side="right", padx=10)
            
            ok_clicked = False
            
            def on_ok():
                nonlocal ok_clicked, selected
                ok_clicked = True
                selected = [var.get() for cb, var, channel in checkboxes if var.get()]
                channel_popup.destroy()
            
            def on_cancel():
                nonlocal selected
                selected = []
                channel_popup.destroy()
            
            tk.Button(ok_cancel_frame, text="OK", command=on_ok).pack(side="left", padx=5)
            tk.Button(ok_cancel_frame, text="Cancel", command=on_cancel).pack(side="left")
            
            # Wait for window to close
            channel_popup.transient(popup)
            channel_popup.grab_set()
            popup.wait_window(channel_popup)
            
            return selected if ok_clicked else None

        # Visualization function
        def visualize_edf_graph():
            selected_file = listbox.get(tk.ACTIVE)
            if not selected_file:
                messagebox.showerror("Error", "Please select an EDF file first")
                return
                
            file_path = os.path.join(edf_data_path, selected_file)
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True)
                channels = raw.ch_names
                selected_channels = select_channels(channels)
                
                # Only proceed if OK was clicked and channels were selected
                if selected_channels is None:
                    return  # User clicked Cancel
                if not selected_channels:
                    messagebox.showwarning("Warning", "No channels selected")
                    return
                    
                plot_line_graph(raw, selected_channels)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read EDF file: {str(e)}")

        # Plotting function
        def plot_line_graph(raw, channels):
            graph_popup = tk.Toplevel(popup)
            graph_popup.title(f"EDF Viewer - {len(channels)} Channels")
            graph_popup.state('zoomed')
            
            # Create a main frame that will hold everything
            main_frame = tk.Frame(graph_popup)
            main_frame.pack(fill="both", expand=True)
            
            # Create a canvas for scrolling
            canvas = tk.Canvas(main_frame)
            canvas.pack(side="left", fill="both", expand=True)
            
            # Add a scrollbar
            scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollbar.pack(side="right", fill="y")
            
            # Configure the canvas
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            
            # Create another frame inside the canvas to hold the plots
            plot_frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=plot_frame, anchor="nw")
            
            # Create matplotlib figure with appropriate size
            fig, axes = plt.subplots(len(channels), 1, 
                                figsize=(12, max(2*len(channels), 10)),  # Minimum height of 10
                                squeeze=False)
            fig.suptitle("EEG Channels", y=1.02)
            
            # Plot each channel
            for idx, channel in enumerate(channels):
                data, times = raw[channel, :]
                axes[idx, 0].plot(times, data.T)
                axes[idx, 0].set_title(channel)
                axes[idx, 0].set_xlabel("Time (s)")
                axes[idx, 0].set_ylabel("Amplitude (μV)")
                axes[idx, 0].grid(True)
            
            plt.tight_layout()
            
            # Embed in Tkinter
            mpl_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            mpl_canvas.draw()
            mpl_canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add toolbar at the bottom (outside the scrollable area)
            toolbar_frame = tk.Frame(graph_popup)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(mpl_canvas, toolbar_frame)
            toolbar.update()
            
            # Make mouse wheel scroll work
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Display button
        tk.Button(popup, text="Visualize Selected File", 
                command=visualize_edf_graph,
                font=("Arial", 10, "bold")).pack(pady=10)

    # Convert edf to csv function
    def convert_csv():
        edf_data_path = os.path.join(patient_path, 'edf_data')
        if not os.path.exists(edf_data_path):
            messagebox.showerror("Error", f"EDF data folder not found")
            return
        
        status_label = tk.Label(left_frame, text="Converting to csv...", fg="#2176BD", font=("Arial", 11))
        status_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
        main_app.update_idletasks()

        convert_csv_button.config(state=tk.DISABLED)

        def convert_csv_thread():
            try:
                csv_data_path = os.path.join(patient_path, 'csv_data')
                os.makedirs(csv_data_path, exist_ok=True)

                files = [f for f in os.listdir(edf_data_path) if f.endswith(".edf")]
                for file in files:
                    edf_file_path = os.path.join(edf_data_path, file)
                    edf_file = mne.io.read_raw_edf(edf_file_path)
                    csv_file_path = os.path.join(csv_data_path, file.replace(".edf", ".csv"))
                    edf_file.to_data_frame().to_csv(csv_file_path, index=False)

                main_app.after(0, lambda: [
                    status_label.destroy(),
                    convert_csv_button.config(state=tk.NORMAL),
                    messagebox.showinfo("Completed", "EDF files successfully converted to CSV")
                ])
            except Exception as e:
                main_app.after(0, lambda: [
                    status_label.destroy(),
                    convert_csv_button.config(state=tk.NORMAL),
                    messagebox.showerror("Error", f"An error occurred in converting files")
                ])
        
        convert_to_csv_thread = threading.Thread(target=convert_csv_thread, daemon=True)
        convert_to_csv_thread.start()
    
    # View csv data function
    def view_csv_data():
        csv_data_path = os.path.join(patient_path, 'csv_data')
        if not os.path.exists(csv_data_path):
            messagebox.showerror("Error", f"CSV data folder not found")
            return

        # List all .csv files in the csv data folder
        files = [f for f in os.listdir(csv_data_path) if f.endswith(".csv")]
        if not files:
            messagebox.showinfo("Error", "No files found in the CSV data folder.")
            return

        # Create a popup window to display the list of CSV files
        popup = tk.Toplevel(main_app)
        popup.title("Select File")
        popup.geometry("400x300")

        top_bar = tk.Frame(popup, bg="#2176BD", height=50)
        top_bar.pack(fill="x", side="top")

        tk.Label(top_bar, text="Select a file to display:", bg="#2176BD", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        listbox = tk.Listbox(popup)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        for file in files:
            listbox.insert(tk.END, file)

        def display_csv():
            selected_file = listbox.get(tk.ACTIVE)
            if selected_file:
                file_path = os.path.join(csv_data_path, selected_file)

                # Open the file in the default CSV viewer
                try:
                    os.startfile(file_path)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to open file")
            else:
                messagebox.showerror("Error", "No file selected")

        display_button = tk.Button(popup, text="Display File", font=("Arial", 10, "bold"), command=display_csv)
        display_button.pack(pady=10)

    # Seizure prediction function
    def predict_seizure():
        status_label = tk.Label(left_frame, text="Localising seizures...", fg="#2176BD", font=("Arial", 11))
        status_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
        main_app.update_idletasks()

        predict_seizure_button.config(state=tk.DISABLED)

        def seizure_prediction_thread():
            try:
                data_path = os.path.join(patient_path, 'eeg_data')
                feature_path = os.path.join(patient_path, 'feature_extraction')
                model_path = os.path.join(patient_path, 'saved_model')
                seizure_time_path = os.path.join(patient_path, 'seizure_time')
                seizure_predictions = PredictSeizure(data_path, feature_path, model_path, seizure_time_path)

                main_app.after(0, lambda: [
                    status_label.destroy(),
                    predict_seizure_button.config(state=tk.NORMAL),
                    messagebox.showinfo("Completed", "Seizures localised successfully.")
                ])
            except Exception as e:
                main_app.after(0, lambda: [
                    status_label.destroy(),
                    predict_seizure_button.config(state=tk.NORMAL),
                    messagebox.showerror("Error", f"An error occurred in localising seizures.")
                ])
        
        predict_seizure_thread = threading.Thread(target=seizure_prediction_thread, daemon=True)
        predict_seizure_thread.start()

    # View seizure times function
    def view_seizures():
        seizure_times_path = os.path.join(patient_path, 'seizure_time')
        if not os.path.exists(seizure_times_path):
            messagebox.showerror("Error", f"Seizure times folder not found")
            return

        # List all .csv files in the seizure times folder
        files = [f for f in os.listdir(seizure_times_path) if f.endswith(".csv")]
        if not files:
            messagebox.showinfo("Error", "No files found in the seizure times folder.")
            return

        # Create a popup window to display the list of CSV files
        popup = tk.Toplevel(main_app)
        popup.title("Select File")
        popup.geometry("400x300")

        top_bar = tk.Frame(popup, bg="#2176BD", height=50)
        top_bar.pack(fill="x", side="top")

        tk.Label(top_bar, text="Select a file to display:", bg="#2176BD", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        listbox = tk.Listbox(popup)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        for file in files:
            listbox.insert(tk.END, file)

        def display_csv():
            selected_file = listbox.get(tk.ACTIVE)
            if selected_file:
                file_path = os.path.join(seizure_times_path, selected_file)

                # Open the file in the default CSV viewer
                try:
                    os.startfile(file_path)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to open file")
            else:
                messagebox.showerror("Error", "No file selected")

        display_button = tk.Button(popup, text="Display File", font=("Arial", 10, "bold"), command=display_csv)
        display_button.pack(pady=10)
    
    # SHAP explainer function
    def generate_shap_explanations():
        status_label = tk.Label(left_frame, text="Generating SHAP explanations...", fg="#2176BD", font=("Arial", 11))
        status_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
        main_app.update_idletasks()
            
        generate_shap_button.config(state=tk.DISABLED)
            
        def shap_explanation_generation_thread():
            #try:
            data_path = os.path.join(patient_path, 'eeg_data')
            model_path = os.path.join(patient_path, r'saved_model\model_checkpoint.keras')
            seizure_time_path = os.path.join(patient_path, 'seizure_time')
            output_path = os.path.join(patient_path, 'shap_explanations')
            shap_explanations = ShapExplainer(data_path, model_path, seizure_time_path, output_path)
            
            main_app.after(0, lambda: [
                status_label.destroy(),
                generate_shap_button.config(state=tk.NORMAL),
                messagebox.showinfo("Completed", "SHAP explanations generated successfully.")
            ])
            '''except Exception as e:
                main_app.after(0, lambda: [
                    status_label.destroy(),
                    generate_shap_explanations_button.config(state=tk.NORMAL),
                    messagebox.showerror("Error", f"An error occurred in generating explanations.")
                ])'''
            
        shap_explanation_thread = threading.Thread(target=shap_explanation_generation_thread, daemon=True)
        shap_explanation_thread.start()

    # View SHAP explanations function
    def view_shap_explanations():        
        shap_explanations_path = os.path.join(patient_path, 'shap_explanations')
        if not os.path.exists(shap_explanations_path):
            messagebox.showerror("Error", f"SHAP results folder not found")
            return

        # List all .png files in the SHAP RESULTS folder
        files = [f for f in os.listdir(shap_explanations_path) if f.endswith(".png")]
        if not files:
            messagebox.showinfo("Error", "No files found in the SHAP results folder")
            return

        # Create a popup window to display the list of PNG files
        popup = tk.Toplevel(main_app)
        popup.title("Select Plot")
        popup.geometry("400x300")

        top_bar = tk.Frame(popup, bg="#2176BD", height=50)
        top_bar.pack(fill="x", side="top")

        tk.Label(top_bar, text="Select a SHAP explanation plot to display:", bg="#2176BD", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        listbox = tk.Listbox(popup)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        for file in files:
            listbox.insert(tk.END, file)

        def display_image():
            selected_file = listbox.get(tk.ACTIVE)
            if selected_file:
                image_path = os.path.join(shap_explanations_path, selected_file)

                # Open the image in a new window
                try:
                    image_window = tk.Toplevel(popup)
                    image_window.title(f"SHAP Explanation: {selected_file}")
                    image_window.geometry("800x600")

                    # Load and display the image
                    img = Image.open(image_path)
                    img = img.resize((750, 550), Image.Resampling.LANCZOS)  # Resize the image
                    img_tk = ImageTk.PhotoImage(img)

                    image_label = tk.Label(image_window, image=img_tk)
                    image_label.image = img_tk  # Keep a reference to avoid garbage collection
                    image_label.pack(pady=10)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to open image")
            else:
                messagebox.showerror("Error", "No file selected")

        display_button = tk.Button(popup, text="Display Image", font=("Arial", 10, "bold"), command=display_image)
        display_button.pack(pady=10)

    # LIME explainer function
    def generate_lime_explanations():
        status_label = tk.Label(left_frame, text="Generating LIME explanations...", fg="#2176BD", font=("Arial", 11))
        status_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
        main_app.update_idletasks()
            
        generate_lime_button.config(state=tk.DISABLED)
            
        def lime_explanation_generation_thread():
            try:
                data_path = os.path.join(patient_path, 'eeg_data')
                model_path = os.path.join(patient_path, r'saved_model\model_checkpoint.keras')
                seizure_time_path = os.path.join(patient_path, 'seizure_time')
                output_path = os.path.join(patient_path, 'lime_explanations')
                lime_explanations = LimeExplainer(data_path, model_path, seizure_time_path, output_path)
                
                main_app.after(0, lambda: [
                    status_label.destroy(),
                    generate_lime_button.config(state=tk.NORMAL),
                    messagebox.showinfo("Completed", "LIME explanations generated successfully.")
                ])
            except Exception as e:
                main_app.after(0, lambda: [
                    status_label.destroy(),
                    generate_lime_button.config(state=tk.NORMAL),
                    messagebox.showerror("Error", f"An error occurred in generating explanations.")
                ])
            
        lime_explanation_thread = threading.Thread(target=lime_explanation_generation_thread, daemon=True)
        lime_explanation_thread.start()

    # View LIME explanations function
    def view_lime_explanations():
        lime_explanations_path = os.path.join(patient_path, 'lime_explanations')
        if not os.path.exists(lime_explanations_path):
            messagebox.showerror("Error", f"LIME explanations folder not found")
            return

        # List all subfolders in the LIME RESULTS folder
        subfolders = [f for f in os.listdir(lime_explanations_path) if os.path.isdir(os.path.join(lime_explanations_path, f))]
        if not subfolders:
            messagebox.showinfo("Error", "No LIME explanations found")
            return

        # Create a popup window to display the list of subfolders
        popup = tk.Toplevel(main_app)
        popup.title("Select Subfolder")
        popup.geometry("400x300")

        top_bar = tk.Frame(popup, bg="#2176BD", height=50)
        top_bar.pack(fill="x", side="top")

        tk.Label(top_bar, text="Select a subfolder to display files:", bg="#2176BD", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        listbox = tk.Listbox(popup)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        for subfolder in subfolders:
            listbox.insert(tk.END, subfolder)

        def display_subfolder_images():
            selected_subfolder = listbox.get(tk.ACTIVE)
            if selected_subfolder:
                subfolder_path = os.path.join(lime_explanations_path, selected_subfolder)

                # List all .png files in the selected subfolder
                files = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]
                if not files:
                    messagebox.showinfo("Error", "No LIME explanations in the selected subfolder")
                    return

                # Create a new popup window to display the list of PNG files
                image_popup = tk.Toplevel(popup)
                image_popup.title(f"Files in {selected_subfolder}")
                image_popup.geometry("400x300")

                image_top_bar = tk.Frame(image_popup, bg="#2176BD", height=50)
                image_top_bar.pack(fill="x", side="top")

                tk.Label(image_top_bar, text="Select a plot to display:", bg="#2176BD", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

                image_listbox = tk.Listbox(image_popup)
                image_listbox.pack(fill="both", expand=True, padx=10, pady=10)

                for file in files:
                    image_listbox.insert(tk.END, file)

                def display_image():
                    selected_file = image_listbox.get(tk.ACTIVE)
                    if selected_file:
                        image_path = os.path.join(subfolder_path, selected_file)

                        # Open the image in a new window
                        try:
                            image_window = tk.Toplevel(image_popup)
                            image_window.title(f"LIME Result: {selected_file}")
                            image_window.geometry("800x600")

                            # Load and display the image
                            img = Image.open(image_path)
                            img = img.resize((750, 550), Image.Resampling.LANCZOS)  # Resize the image
                            img_tk = ImageTk.PhotoImage(img)

                            image_label = tk.Label(image_window, image=img_tk)
                            image_label.image = img_tk  # Keep a reference to avoid garbage collection
                            image_label.pack(pady=10)
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to open image")
                    else:
                        messagebox.showerror("Error", "No file selected")

                display_button = tk.Button(image_popup, text="Display Plot", font=("Arial", 10, "bold"), command=display_image)
                display_button.pack(pady=10)
            else:
                messagebox.showerror("Error", "No subfolder selected.")

        display_button = tk.Button(popup, text="Open Subfolder", font=("Arial", 10, "bold"), command=display_subfolder_images)
        display_button.pack(pady=10)

    # Main content frames
    left_frame = tk.Frame(main_app, padx=20, pady=20)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = tk.Frame(main_app, padx=20, pady=20)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # RIGHT SIDE - Patient Details
    patient_details_frame = tk.LabelFrame(right_frame, text="Patient Details", padx=10, pady=10)
    patient_details_frame.pack(fill=tk.BOTH, expand=True)

    # Create entry widgets as instance variables
    global name_entry, age_entry, notes_entry, patient_path
    patient_path = ""

    # Name
    tk.Label(patient_details_frame, text="Name:", font=('Arial', 11)).pack(anchor='w')
    name_entry = tk.Entry(patient_details_frame, width=35, font=('Arial', 11))
    name_entry.pack(fill=tk.X, pady=5)

    # Age
    tk.Label(patient_details_frame, text="Age:", font=('Arial', 11)).pack(anchor='w')
    age_entry = tk.Entry(patient_details_frame, width=35, font=('Arial', 11))
    age_entry.pack(fill=tk.X, pady=5)

    # Notes
    tk.Label(patient_details_frame, text="Notes:", font=('Arial', 11)).pack(anchor='w')
    notes_entry = tk.Text(patient_details_frame, height=5, width=35, font=('Arial', 11))
    notes_entry.pack(fill=tk.X, pady=5)

    # Patient action buttons
    button_frame = tk.Frame(patient_details_frame)
    button_frame.pack(fill=tk.X, pady=(15, 5))

    select_patient_button = tk.Button(button_frame, text="Select Patient", 
                                    font=('Arial', 11, 'bold'), padx=15, pady=5,
                                    command=select_patient)
    select_patient_button.pack(side=tk.LEFT, padx=5, expand=True)

    update_patient_button = tk.Button(button_frame, text="Save Details", 
                                    font=('Arial', 11, 'bold'), padx=15, pady=5,
                                    command=update_patient_details)
    update_patient_button.pack(side=tk.RIGHT, padx=5, expand=True)

    # LEFT SIDE - Analysis Tools
    tools_frame = tk.LabelFrame(left_frame, text="Analysis Tools", padx=10, pady=10)
    tools_frame.pack(fill=tk.BOTH, expand=True)

    # View edf data button
    view_edf_frame = tk.Frame(tools_frame)
    view_edf_frame.pack(fill=tk.X, pady=5)

    view_edf_button = tk.Button(view_edf_frame, text="View EDF Data", 
                                     font=('Arial', 11, 'bold'), padx=15, pady=8,
                                     command=view_edf_data)
    view_edf_button.pack(side=tk.LEFT, padx=5, expand=True)

    # Manage csv data buttons (same row)
    csv_data_frame = tk.Frame(tools_frame)
    csv_data_frame.pack(fill=tk.X, pady=5)

    convert_csv_button = tk.Button(csv_data_frame, text="Convert EDF to CSV", 
                                  font=('Arial', 11, 'bold'), padx=15, pady=8,
                                  command=convert_csv)
    convert_csv_button.pack(side=tk.RIGHT, padx=5, expand=True)

    view_csv_button = tk.Button(csv_data_frame, text="View CSV Data", 
                                     font=('Arial', 11, 'bold'), padx=15, pady=8,
                                     command=view_csv_data)
    view_csv_button.pack(side=tk.LEFT, padx=5, expand=True)
    
    # Seizure buttons (same row)
    seizure_frame = tk.Frame(tools_frame)
    seizure_frame.pack(fill=tk.X, pady=5)

    predict_seizure_button = tk.Button(seizure_frame, text="Localise Seizures", 
                                     font=('Arial', 11, 'bold'), padx=15, pady=8,
                                     command=predict_seizure)
    predict_seizure_button.pack(side=tk.LEFT, padx=5, expand=True)

    view_seizure_button = tk.Button(seizure_frame, text="View Seizures", 
                                  font=('Arial', 11, 'bold'), padx=15, pady=8,
                                  command=view_seizures)
    view_seizure_button.pack(side=tk.RIGHT, padx=5, expand=True)

    # SHAP buttons (same row)
    shap_frame = tk.Frame(tools_frame)
    shap_frame.pack(fill=tk.X, pady=5)

    generate_shap_button = tk.Button(shap_frame, text="SHAP Explainer", 
                                   font=('Arial', 11, 'bold'), padx=15, pady=8,
                                   command=generate_shap_explanations)
    generate_shap_button.pack(side=tk.LEFT, padx=5, expand=True)

    view_shap_button = tk.Button(shap_frame, text="SHAP Results", 
                               font=('Arial', 11, 'bold'), padx=15, pady=8,
                               command=view_shap_explanations)
    view_shap_button.pack(side=tk.RIGHT, padx=5, expand=True)

    # LIME buttons (same row)
    lime_frame = tk.Frame(tools_frame)
    lime_frame.pack(fill=tk.X, pady=5)

    generate_lime_button = tk.Button(lime_frame, text="LIME Explainer", 
                                   font=('Arial', 11, 'bold'), padx=15, pady=8,
                                   command=generate_lime_explanations)
    generate_lime_button.pack(side=tk.LEFT, padx=5, expand=True)

    view_lime_button = tk.Button(lime_frame, text="LIME Results", 
                               font=('Arial', 11, 'bold'), padx=15, pady=8,
                               command=view_lime_explanations)
    view_lime_button.pack(side=tk.RIGHT, padx=5, expand=True)

    main_app.mainloop()

# Create the login screen
login_screen = tk.Tk()
login_screen.title("Login")
login_screen.state('zoomed')

# Create a main frame that will hold both image and login components
main_frame = tk.Frame(login_screen)
main_frame.pack(fill=tk.BOTH, expand=True)

# Configure grid weights for 2:1 ratio
main_frame.grid_columnconfigure(0, weight=2)  # Left side (image) gets 2 parts
main_frame.grid_columnconfigure(1, weight=1)  # Right side (login) gets 1 part
main_frame.grid_rowconfigure(0, weight=1)     # Single row

# Left side - Image (takes 2/3 width)
left_frame = tk.Frame(main_frame, bg='white')
left_frame.grid(row=0, column=0, sticky='nsew')

# Load and display the image
original_img = Image.open("img1.jpg")

# Create canvas for the image
img_canvas = tk.Canvas(left_frame, bg='white', highlightthickness=0)
img_canvas.pack(fill=tk.BOTH, expand=True)

# Function to resize and display the image
def display_image(event=None):
    # Get current canvas dimensions
    canvas_width = img_canvas.winfo_width()
    canvas_height = img_canvas.winfo_height()
    
    if canvas_width > 0 and canvas_height > 0:
        # Resize image to fit canvas while maintaining aspect ratio
        resized_img = ImageOps.fit(original_img, (canvas_width, canvas_height), 
                                 method=Image.LANCZOS, centering=(0.5, 0.5))
        img_tk = ImageTk.PhotoImage(resized_img)
        
        # Update the image on canvas
        img_canvas.delete("all")
        img_canvas.create_image(canvas_width//2, canvas_height//2, 
                               anchor=tk.CENTER, image=img_tk)
        img_canvas.image = img_tk  # Keep reference

# Initial display
display_image()

# Bind resize event
img_canvas.bind('<Configure>', display_image)

# Right side - Login form (takes 1/3 width)
right_frame = tk.Frame(main_frame, padx=50)
right_frame.grid(row=0, column=1, sticky='nsew')

# Welcome text
welcome_font = tkfont.Font(size=24, weight='bold')
welcome_label = tk.Label(right_frame, text="Welcome", font=welcome_font)
welcome_label.pack(pady=(100, 50))  # More padding at top

# Username components
username_frame = tk.Frame(right_frame)
username_frame.pack(pady=10, fill=tk.X)

username_label = tk.Label(username_frame, text="Username:")
username_label.pack(anchor=tk.W)

username_entry = tk.Entry(username_frame, width=30, font=('Arial', 12))
username_entry.pack(fill=tk.X, pady=5, ipady=8)  # ipady makes the entry taller

# Password components
password_frame = tk.Frame(right_frame)
password_frame.pack(pady=10, fill=tk.X)

password_label = tk.Label(password_frame, text="Password:")
password_label.pack(anchor=tk.W)

password_entry = tk.Entry(password_frame, show="*", width=30, font=('Arial', 12))
password_entry.pack(fill=tk.X, pady=5, ipady=8)  # ipady makes the entry taller

# Login button
login_button = tk.Button(right_frame, text="Login", command=login, 
                        font=('Arial', 12), padx=20, pady=10)
login_button.pack(pady=30)

login_screen.mainloop()
