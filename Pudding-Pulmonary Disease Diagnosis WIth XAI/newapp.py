import sys
import sqlite3
import torch.nn as nn
import torch
import os
import cv2
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                               QLabel, QLineEdit, QStackedWidget, QFormLayout,
                               QMessageBox, QListWidget, QDialog, QComboBox, QFileDialog, QScrollArea)
from PySide6.QtGui import QFont, QDesktopServices, QPixmap, QImage
from PySide6.QtCore import QStandardPaths, Qt, QUrl
from fpdf import FPDF
from torchvision import transforms, models
from PIL import Image
from gradcam import GradCAM, visualize_gradcam
from CustomCNN import CustomCNN
from shap1 import SHAPExplainer
from lime1 import DLIMEExplainer

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pulmonary Disease Detection System")
        self.setGeometry(300, 150, 700, 500)

        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomCNN(num_classes=3)  # Ensure the model structure matches saved weights
        self.model.load_state_dict(torch.load("customModelDictCOVID.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load CNN Model and SHAP Explainer
        self.shap_explainer = SHAPExplainer()

        # Apply styles
        self.setStyleSheet("""
            QWidget {
                background-color: #2C2F33;
                color: #F1F1F1;
                font-family: Arial;
            }
            QLabel {
                color: #F1F1F1;
                font-size: 16px;
                padding: 5px 0;
            }
            QLineEdit, QComboBox {
                border: 1px solid #7289DA;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                background-color: #23272A;
                color: #F1F1F1;
            }
            QPushButton {
                color: white;
                background-color: #7289DA;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5B6EAE;
            }
            QListWidget {
                border: 1px solid #7289DA;
                font-size: 14px;
                background-color: #23272A;
                color: #F1F1F1;
            }
        """)

        # Database setup
        self.db_connection = sqlite3.connect("app_database.db")
        self.create_db()

        # Stack for different pages
        self.stacked_widget = QStackedWidget(self)

        # Create pages
        self.create_login_page()
        self.create_create_account_page()
        self.create_patient_management_page()

        # Set initial page to login page
        self.stacked_widget.setCurrentIndex(0)

        # Layout for the main window
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked_widget)

    def create_db(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS doctors (username TEXT PRIMARY KEY, password TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS patient_records (id TEXT PRIMARY KEY, name TEXT, age NUMBER, gender TEXT, diagnosis TEXT)''')
        self.db_connection.commit()

    def create_login_page(self):
        self.login_page = QWidget()
        login_layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        login_layout.addWidget(QLabel("Username"))
        login_layout.addWidget(self.username_input)
        login_layout.addWidget(QLabel("Password"))
        login_layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.handle_login)
        login_layout.addWidget(self.login_button)

        self.create_account_button = QPushButton("Create Account")
        self.create_account_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        login_layout.addWidget(self.create_account_button)

        self.login_page.setLayout(login_layout)
        self.stacked_widget.addWidget(self.login_page)

    def create_create_account_page(self):
        self.create_account_page = QWidget()
        create_account_layout = QFormLayout()

        self.new_username_input = QLineEdit()
        self.new_password_input = QLineEdit()
        self.new_password_input.setEchoMode(QLineEdit.Password)

        create_account_layout.addRow("New Username", self.new_username_input)
        create_account_layout.addRow("New Password", self.new_password_input)

        self.create_account_btn = QPushButton("Create Account")
        self.create_account_btn.clicked.connect(self.handle_create_account)
        create_account_layout.addWidget(self.create_account_btn)

        self.back_to_login_button = QPushButton("Back to Login")
        self.back_to_login_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        create_account_layout.addWidget(self.back_to_login_button)

        self.create_account_page.setLayout(create_account_layout)
        self.stacked_widget.addWidget(self.create_account_page)

    def create_patient_management_page(self): 
        self.patient_management_page = QWidget()
        patient_layout = QVBoxLayout()

        self.add_patient_button = QPushButton("Add Patient")
        self.add_patient_button.clicked.connect(self.open_add_patient_window)
        patient_layout.addWidget(self.add_patient_button)

        self.view_patient_button = QPushButton("View Existing Patient Record")
        self.view_patient_button.clicked.connect(self.show_patient_list)
        patient_layout.addWidget(self.view_patient_button)

        self.patient_list_widget = QListWidget()
        self.patient_list_widget.itemClicked.connect(self.show_patient_details)
        patient_layout.addWidget(self.patient_list_widget)

        self.patient_details_label = QLabel()
        patient_layout.addWidget(self.patient_details_label)

        self.delete_patient_button = QPushButton("Delete Patient")
        self.delete_patient_button.clicked.connect(self.delete_patient)
        patient_layout.addWidget(self.delete_patient_button)

        self.delete_all_patients_button = QPushButton("Delete All Patients")
        self.delete_all_patients_button.clicked.connect(self.delete_all_patients)
        patient_layout.addWidget(self.delete_all_patients_button)

        # Add an image upload button
        self.upload_button = QPushButton("Upload X-ray Image")
        self.upload_button.clicked.connect(self.upload_image)
        patient_layout.addWidget(self.upload_button)

        # Label to show predicted result
        self.result_label = QLabel("Prediction: ")
        patient_layout.addWidget(self.result_label)

        self.gradcam_button = QPushButton("Explain with GradCAM")
        self.gradcam_button.clicked.connect(self.run_gradcam)
        patient_layout.addWidget(self.gradcam_button)

        self.shap_button = QPushButton("Explain with SHAP")
        self.shap_button.clicked.connect(self.run_shap)
        patient_layout.addWidget(self.shap_button)

        # self.dlime_button = QPushButton("Explain with DLIME")
        # self.dlime_button.clicked.connect(self.run_dlime)
        # patient_layout.addWidget(self.dlime_button)

        self.explanation_label = QLabel("Explanation")
        self.explanation_label.setAlignment(Qt.AlignCenter)  # Center align the text
        patient_layout.addWidget(self.explanation_label)

        self.download_report_button = QPushButton("Download Report (PDF)")
        self.download_report_button.clicked.connect(self.download_patient_report)
        self.download_report_button.setVisible(False)
        patient_layout.addWidget(self.download_report_button)

        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)
        patient_layout.addWidget(self.logout_button)

        # Wrap in scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_content.setLayout(patient_layout)

        scroll_area.setWidget(scroll_content)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)

        self.patient_management_page.setLayout(main_layout)
        self.stacked_widget.addWidget(self.patient_management_page)

    def open_add_patient_window(self):
        self.add_patient_window = AddPatientWindow(self.db_connection)
        self.add_patient_window.exec()

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM doctors WHERE username = ? AND password = ?", (username, password))
        if cursor.fetchone():
           self.stacked_widget.setCurrentIndex(2)  # Switch to patient management page
           QMessageBox.information(self, "Login Successful", "Welcome to the Pulmonary Disease Detection System!")  # Success message
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect username or password")

    def handle_create_account(self):
        username = self.new_username_input.text()
        password = self.new_password_input.text()
        cursor = self.db_connection.cursor()
        try:
            cursor.execute("INSERT INTO doctors (username, password) VALUES (?, ?)", (username, password))
            self.db_connection.commit()
            QMessageBox.information(self, "Account Created", "Account created successfully")
            self.stacked_widget.setCurrentIndex(0)
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Error", "Username already exists")

    def show_patient_list(self):
        self.patient_list_widget.clear()
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT id, name FROM patient_records")
        patients = cursor.fetchall()
        for patient in patients:
            self.patient_list_widget.addItem(f"{patient[0]} - {patient[1]}")

    def show_patient_details(self, item):
        patient_id = item.text().split(" - ")[0]
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM patient_records WHERE id = ?", (patient_id,))
        patient = cursor.fetchone()
        if patient:
            self.patient_details_label.setText(f"ID: {patient[0]}\nName: {patient[1]}\nAge: {patient[2]}\nGender: {patient[3]}\nDiagnosis: {patient[4]}")
            self.download_report_button.setVisible(True)

    def delete_patient(self):
        current_item = self.patient_list_widget.currentItem()
        if current_item:
            patient_id = current_item.text().split(" - ")[0]
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM patient_records WHERE id = ?", (patient_id,))
            self.db_connection.commit()
            self.show_patient_list()
            self.patient_details_label.clear()
            QMessageBox.information(self, "Success", "Patient record deleted successfully")

    def delete_all_patients(self):
        confirm = QMessageBox.question(self, "Confirmation", "Are you sure you want to delete all patients?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM patient_records")
            self.db_connection.commit()
            self.show_patient_list()
            self.patient_details_label.clear()

    def upload_image(self):
        if not self.patient_details_label.text():  # Check if a patient is selected
            QMessageBox.warning(self, "Error", "Please select a patient before uploading an X-ray image.")
            return
        
        self.current_image, _ = QFileDialog.getOpenFileName(self, "Select X-ray Image", "", "Images (*.png *.jpg *.jpeg)")
        if self.current_image:
            self.prediction = self.predict_disease(self.current_image)
            self.result_label.setText(f"Prediction: {self.prediction}")

    def predict_disease(self, image_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert("L")  # Convert to grayscale if needed
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Move image tensor to the same device as the model
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        
        self.predicted = predicted
        classes = ["COVID-19", "Normal", "Pneumonia"]  # Update based on your model
        prediction_result = classes[predicted.item()]

        return prediction_result
    
    def show_gradcam_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.explanation_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_shap_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.explanation_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_shap(self):
        # Extract patient ID from selected item
        current_item = self.patient_list_widget.currentItem()
        patient_id = current_item.text().split(" - ")[0]
        patient_name = current_item.text().split(" - ")[1]

        # Update the patient's diagnosis in the database
        cursor = self.db_connection.cursor()
        cursor.execute("UPDATE patient_records SET diagnosis = ? WHERE id = ?", (self.prediction, patient_id))
        self.db_connection.commit()

        # Save SHAP explanation
        save_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save SHAP Output")
        shap_path = os.path.join(save_folder, f"{patient_id}_{patient_name}_shap_output.png") if save_folder else f"{patient_id}_{patient_name}_shap_output.png"
        shap_path = self.shap_explainer.explain(self.current_image, shap_path)

        self.show_shap_image(shap_path)

        # Refresh UI with updated details
        self.show_patient_list()  # Refresh list
        self.show_patient_details(self.patient_list_widget.currentItem())  # Refresh patient details

        QMessageBox.information(self, "Diagnosis Updated", f"Diagnosis for Patient ID {patient_id} updated to {self.prediction}")

    def run_gradcam(self):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(self.current_image).convert("RGB")  # Convert to grayscale if needed
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Move image tensor to the same device as the model
        image = image.to(self.device)

        # Extract patient ID and name for naming
        current_item = self.patient_list_widget.currentItem()
        patient_id = current_item.text().split(" - ")[0]
        patient_name = current_item.text().split(" - ")[1]

        # Apply Grad-CAM
        target_layer = self.model.conv3    # Last convolutional layer
        gradcam = GradCAM(self.model, target_layer)
        heatmap = gradcam.generate(image, self.predicted.item())

        save_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Grad-CAM")
        if save_folder:
            gradcam_path = os.path.join(save_folder, f"{patient_id}_{patient_name}_gradcam_output.png")
        else:
            gradcam_path =  f"{patient_id}_{patient_name}_gradcam_output.png"  # Default path if user cancels
        visualize_gradcam(heatmap, image.cpu(), gradcam_path)

        # Display Grad-CAM result in UI
        self.show_gradcam_image(gradcam_path)

    # def run_dlime(self):
    #     if self.current_image is None or self.model is None:
    #         print("No image or model loaded")
    #         return
        
    #     explainer = DLIMEExplainer()
    #     xray_image = cv2.imread(self.current_image, cv2.IMREAD_GRAYSCALE)

    #     explainer.explain(xray_image, self.model)

    def download_patient_report(self):
        if not self.patient_details_label.text():  # Check if a patient is selected
            QMessageBox.warning(self, "Error", "Please select a patient before downloading the report.")
            return

        # Get the currently selected patient
        current_item = self.patient_list_widget.currentItem()
        if not current_item:  # Additional check if no patient is selected
            QMessageBox.warning(self, "Error", "No patient selected. Please select a patient first.")
            return

        patient_id = current_item.text().split(" - ")[0]
        patient_name = current_item.text().split(" - ")[1]

        pdf = FPDF()
        pdf.add_page()
    
    # Header for the report
        pdf.set_font("Arial", "B", 20)
        pdf.set_text_color(30, 144, 255)  # Dodger blue color
        pdf.cell(0, 15, txt="Pulmonary Disease Detection System", ln=True, align="C")
        pdf.set_text_color(0, 0, 0)  # Reset to black
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, txt="Patient Diagnosis Report", ln=True, align="C")
        pdf.ln(10)

    # Subtitle with Patient Information header
        pdf.set_fill_color(240, 240, 240)  # Light gray background for section header
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Patient Information", ln=True, fill=True)
        pdf.ln(5)

    # Details with a border and subtle background color
        pdf.set_fill_color(245, 245, 245)
        pdf.set_font("Arial", size=12)
        details = self.patient_details_label.text().splitlines()
        for line in details:
            if "Name:" in line:
                patient_name = line.split("Name:")[1].strip()  # Extract patient's name for filename
            pdf.cell(0, 10, txt=line, ln=True, fill=True, border=1)
        pdf.ln(5)

    # Diagnosis Details Section
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="Diagnosis Details", ln=True, fill=True)
        pdf.ln(5)

    # Diagnosis details with border and background color
        pdf.set_fill_color(245, 245, 245)
        pdf.set_font("Arial", size=12)
        for line in details:
            if "Diagnosis:" in line:
                pdf.cell(0, 10, txt=f"Diagnosis: {self.prediction}", ln=True, fill=True, border=1)
        pdf.ln(10)

        # Check if we have an X-ray image to include
        if hasattr(self, 'current_image') and self.current_image:
            # Original X-ray Image section
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, txt="Original X-ray Image", ln=True, fill=True)
            pdf.ln(5)
            
            try:
                # Add original X-ray image (resized to fit)
                pdf.image(self.current_image, x=50, w=110, h=110)
                pdf.ln(120)
            except Exception as e:
                pdf.cell(0, 10, txt=f"Could not include X-ray image: {str(e)}", ln=True)
        
        # XAI Explanations section
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, txt="AI Explanation Visualizations", ln=True, fill=True)
        pdf.ln(5)
        
        # Check for Grad-CAM explanation
        gradcam_path = f"{patient_id}_{patient_name}_gradcam_output.png"
        if os.path.exists(gradcam_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt="Grad-CAM Explanation:", ln=True)
            pdf.ln(5)
            try:
                pdf.image(gradcam_path, x=30, w=150, h=150)
                pdf.ln(160)
            except Exception as e:
                pdf.cell(0, 10, txt=f"Could not include Grad-CAM image: {str(e)}", ln=True)
        
        # Check for SHAP explanation
        shap_path = f"{patient_id}_{patient_name}_shap_output.png"
        if os.path.exists(shap_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt="SHAP Explanation:", ln=True)
            pdf.ln(5)
            try:
                pdf.image(shap_path, x=30, w=150, h=150)
                pdf.ln(160)
            except Exception as e:
                pdf.cell(0, 10, txt=f"Could not include SHAP image: {str(e)}", ln=True)
        
        # Check for DLIME explanation (assuming it's saved similarly)
        dlime_path = f"{patient_id}_{patient_name}_dlime_output.png"
        if os.path.exists(dlime_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt="DLIME Explanation:", ln=True)
            pdf.ln(5)
            try:
                pdf.image(dlime_path, x=30, w=150, h=150)
                pdf.ln(160)
            except Exception as e:
                pdf.cell(0, 10, txt=f"Could not include DLIME image: {str(e)}", ln=True)

    # Footer for the report
        pdf.set_y(-15)  # Position footer 15mm from bottom
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "Generated by Pulmonary Disease Detection System", align="C")

    # Saving PDF with a formatted filename in the Documents directory
        patient_id = self.patient_list_widget.currentItem().text().split(" - ")[0]
        pdf_output = os.path.join(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation), f"{patient_name}_{patient_id}_Diagnosis_Report.pdf")
        pdf.output(pdf_output)

    # Open the PDF file using QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_output))

    # Success message with the file path
        QMessageBox.information(self, "PDF Downloaded", f"Report saved to {pdf_output}")
    
    def logout(self):
        self.stacked_widget.setCurrentIndex(0)
        self.username_input.clear()  # Clear the username field
        self.password_input.clear()

class AddPatientWindow(QDialog):
    def __init__(self, db_connection):
        super().__init__()
        self.setWindowTitle("Add Patient")
        self.setGeometry(400, 200, 300, 250)
        self.db_connection = db_connection

        self.setStyleSheet("""
            QDialog {
                background-color: #2C2F33;
                color: #F1F1F1;
                font-family: Arial;
            }
            QLabel {
                color: #F1F1F1;
                font-size: 16px;
                padding: 5px 0;
            }
            QLineEdit, QComboBox {
                border: 1px solid #7289DA;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                background-color: #23272A;
                color: #F1F1F1;
                margin-bottom: 10px;
            }
            QPushButton {
                color: white;
                background-color: #7289DA;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 15px;
                font-weight: bold;
                margin-top: 15px;
            }
            QPushButton:hover {
                background-color: #5B6EAE;
            }
        """)

        layout = QFormLayout()

        self.id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.age_input = QLineEdit()
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female", "Other"])
        self.diagnosis_input = QLineEdit()

        layout.addRow("ID", self.id_input)
        layout.addRow("Name", self.name_input)
        layout.addRow("Age", self.age_input)
        layout.addRow("Gender", self.gender_combo)
        layout.addRow("Diagnosis", self.diagnosis_input)

        add_button = QPushButton("Add Patient")
        add_button.clicked.connect(self.add_patient_to_db)
        layout.addWidget(add_button)

        self.setLayout(layout)

    def add_patient_to_db(self):
        patient_id = self.id_input.text()
        name = self.name_input.text()
        age = self.age_input.text()
        gender = self.gender_combo.currentText()  # Ensure this is the correct field for gender
        diagnosis = self.diagnosis_input.text()
        print(patient_id)
        print(name)
        print(age)
        print(gender)
        print(diagnosis)
        cursor = self.db_connection.cursor()
        try:
            cursor.execute('''INSERT INTO patient_records (id, name, age, gender, diagnosis) VALUES (?, ?, ?, ?, ?)''',
                           (patient_id, name, age, gender, diagnosis))
            self.db_connection.commit()
            QMessageBox.information(self, "Success", "Patient added successfully")
            self.accept()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Error", "Patient ID already exists")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()   
    sys.exit(app.exec())