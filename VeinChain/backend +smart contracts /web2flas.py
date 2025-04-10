from flask import Flask, request, jsonify
from web3 import Web3
import json
from Crypto.Cipher import AES
#import base64
from Crypto.Util.Padding import pad ,unpad
from flasgger import Swagger ,swag_from
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
import io
from feature_extractor import extract_feature
import ssl
import gc
import os
import time
from flask_cors import CORS
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
# Run the function
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  
swagger=Swagger(app)
# Connect to Ganache (or another Ethereum node)
ganache_url = os.getenv("WEB3_PROVIDER")  # Ganache default URL
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Check if connected to the network
if not web3.is_connected():
    print("Failed to connect to the network.")
    exit()

# Contract ABI (replace with actual ABI)
contract_abi = json.loads('''
 [
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "string",
          "name": "uid",
          "type": "string"
        },
        {
          "indexed": false,
          "internalType": "bool",
          "name": "exists",
          "type": "bool"
        },
        {
          "indexed": false,
          "internalType": "bytes",
          "name": "featureVector",
          "type": "bytes"
        }
      ],
      "name": "UserAuthenticated",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "string",
          "name": "uid",
          "type": "string"
        },
        {
          "indexed": true,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        }
      ],
      "name": "UserEnrolled",
      "type": "event"
    },
    {
      "inputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "name": "biometricTemplates",
      "outputs": [
        {
          "internalType": "string",
          "name": "encryptedFeatureVector",
          "type": "string"
        },
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [
        {
          "internalType": "string",
          "name": "uid",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "encryptedFeatureVector",
          "type": "string"
        }
      ],
      "name": "enrollUser",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "string",
          "name": "uid",
          "type": "string"
        }
      ],
      "name": "authenticateUser",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        },
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    }
  ]''')

contract_address = os.getenv("CONTRACT_ADDRESS")  # Replace with your contract address
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Set up your account and private key
account = os.getenv("ACCOUNT")

private_key = os.getenv("PRIVATE_KEY")  # Replace with your private key
aes_key = bytes.fromhex(os.getenv("AES_KEY"))
aes_iv = bytes.fromhex(os.getenv("AES_IV"))

def xor_encrypt(data, key):
    scaled_data = np.array(data, dtype=np.uint)  # Enforce 8-bit integers
    print("Scaled Data:", scaled_data)  # Debugging output
    x=bytes([scaled_data[i] ^ key[i % len(key)] for i in range(len(scaled_data))])
    print(x)
    return x.hex()
def float_vector_to_bytes(vector):
    return np.array(vector, dtype=np.float32).flatten().tobytes()

def bytes_to_float_vector(byte_data):
    return np.frombuffer(byte_data, dtype=np.float32).flatten()

# AES encyrp method (keey keep same for encrypt and decrypt)

def encrypt_aes(data):
    cipher = AES.new(aes_key, AES.MODE_CBC, aes_iv)
    encrypted = cipher.encrypt(pad(data, AES.block_size))
    return encrypted.hex()

def decrypt_aes(encrypted_data):
    cipher = AES.new(aes_key, AES.MODE_CBC, aes_iv)
    decrypted = unpad(cipher.decrypt(bytes.fromhex(encrypted_data)), AES.block_size)
    return decrypted
def preprocess_image1(image, box=(280, 140, 410, 500), width=128, height=128):
    crop = image.crop(box)
    crop = crop.rotate(90, expand=True)
    crop = crop.resize((width, height))
    
    img_array = np.asarray(crop, dtype=np.uint8)  # Use uint8 (less memory than float32)
    clahe_img = equalize_adapthist(img_array, clip_limit=0.02)  # Reduce clip_limit
    
    return clahe_img


def preprocess_image(image, box=(280, 120, 490, 290), width=128, height=128):
    # Load image
    # Resize to 640x480
    
    # Crop the region (80,170) to (450,290)
    crop = image.crop(box)
    
    # Rotate 90 degrees
    crop = crop.rotate(90, expand=True)
    
    # Resize to 128x128
    crop = crop.resize((width, height))
    
    # Convert to numpy array
    img_array = np.asarray(crop, dtype=np.uint8)  
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)
    
    return clahe_img

# Route 1: Enroll User with UID and encrypted template
@app.route('/enroll', methods=['POST'])
@swag_from({
    "summary": "Enroll a user",
    "tags": ["User Enrollment"],
    "consumes": ["multipart/form-data"],
    "parameters": [
        {
            "name": "uid",
            "in": "formData",
            "required": True,
            "type": "string",
            "example": "user123"
        },
        {
            "name": "image",
            "in": "formData",
            "required": True,
            "type": "file"
        }
    ],
    "responses": {
        "200": {
            "description": "User successfully enrolled",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "User enrolled"},
                    "txHash": {"type": "string", "example": "0xabc123..."}
                }
            }
        },
        "400": {
            "description": "Error occurred",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "example": "No image provided"}
                }
            }
        }
    }
})
def enroll_user():
    start_time = time.time()
    try:
        uid = request.form['uid']
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))

        processed_image = preprocess_image(image)
        feature_vector = extract_feature(processed_image)  # 128 floating point values

        feature_bytes = float_vector_to_bytes(feature_vector)
        print("feature vector:",feature_vector.tolist())
        print("feature_bytes",feature_bytes)  # Convert float to bytes
        encrypted_template = encrypt_aes(feature_bytes)  # Encrypt bytes
        '''if isinstance(encrypted_template, str):  
          encrypted_template = encrypted_template.encode('utf-8')  # Ensure it's bytes'''
        print("encrypted_template:", encrypted_template)
        transaction = contract.functions.enrollUser(uid, encrypted_template).build_transaction({
            'chainId': 1337,
            'gas': 6721975,
            'gasPrice': web3.to_wei('20', 'gwei'),
            'nonce': web3.eth.get_transaction_count(account),
        })

        signed_transaction = web3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_transaction.raw_transaction)
        #get_memory_usage()
        receipt=web3.eth.wait_for_transaction_receipt(tx_hash)
       
        del image, feature_vector, feature_bytes  # Free memory
        gc.collect()
        elapsed_time = time.time() -  start_time
        return jsonify({"status": "User enrolled", "txHash": tx_hash.hex(),"feature vector":encrypted_template,"gas": receipt.gasUsed,
            "time": f"{elapsed_time:.4f} seconds"}), 200

    except Exception as e:
        return {"error": str(e)}, 400
  

# Route 2: Authenticate User with UID and query template
@app.route('/authenticate', methods=['POST'])
@swag_from({
    "summary": "Authenticate a user",
    "tags": ["User Authentication"],
    "consumes": ["multipart/form-data"],
    "parameters": [
        {
            "name": "uid",
            "in": "formData",
            "required": True,
            "type": "string",
            "example": "user123"
        },
        {
            "name": "image",
            "in": "formData",
            "required": True,
            "type": "file"
        }
    ],
    "responses": {
        "200": {
            "description": "Authentication result",
            "schema": {
                "type": "object",
                "properties": {
                    "authenticationResult": {"type": "string", "example": "User authenticated successfully"}
                }
            }
        },
        "400": {
            "description": "Error occurred",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "example": "No image provided"}
                }
            }
        }
    }
})
def authenticate_user():
    start_time = time.time()
    
    try:
        uid = request.form['uid']
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))

        processed_image = preprocess_image(image)
        feature_vector = extract_feature(processed_image)
        print("Feature Vector auth :", feature_vector.tolist())
        gas_estimate = contract.functions.authenticateUser(uid).estimate_gas()
        is_enrolled, encrypted_stored_template = contract.functions.authenticateUser(uid).call()

        if not is_enrolled:
            return jsonify({"authenticationResult": "User does not exist"}), 400
        print("encrypted_stored_template:", encrypted_stored_template)
        stored_feature_bytes = decrypt_aes(encrypted_stored_template)  # Decrypt bytes
        stored_feature_vector = bytes_to_float_vector(stored_feature_bytes) 
         # Convert to float vector
        print("stored_feature_bytes:", stored_feature_bytes)
        print("stored_feature_vector:", stored_feature_vector.tolist()) 
        similarity_score = np.dot(feature_vector, stored_feature_vector) / (np.linalg.norm(feature_vector) * np.linalg.norm(stored_feature_vector)) * 100
        print("Similarity score:", similarity_score)
        #get_memory_usage()
        elapsed_time = time.time() - start_time
        if similarity_score >= 86:
            return jsonify({"authenticationResult": f"User:{uid} authenticated successfully","time": f"{elapsed_time:.4f} seconds",
            "gasUsed": gas_estimate}), 200
        else:
            print("hi")
            return jsonify({"authenticationResult": "Authentication failed","time": f"{elapsed_time:.4f} seconds",
            "gasUsed": gas_estimate}), 400

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 400
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    app.run(host="0.0.0.0", port=port,debug=True)
    #app.run(debug=True)
