/**
 * Compresses an image from a video element to a smaller size
 * @param video The video element to capture
 * @param quality The JPEG quality (0-1)
 * @param maxWidth The maximum width of the output image
 * @returns A Promise that resolves to a base64 string of the compressed image
 */
export const captureCompressedImage = (
  video: HTMLVideoElement, 
  quality = 0.6,
  maxWidth = 640
): Promise<string> => {
  return new Promise((resolve, reject) => {
    try {
      // Create a canvas element
      const canvas = document.createElement('canvas');
      
      // Calculate dimensions to maintain aspect ratio
      const aspectRatio = video.videoWidth / video.videoHeight;
      const width = Math.min(video.videoWidth, maxWidth);
      const height = width / aspectRatio;
      
      // Set dimensions
      canvas.width = width;
      canvas.height = height;
      
      // Draw the video frame to the canvas
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }
      
      ctx.drawImage(video, 0, 0, width, height);
      
      // Convert to base64 with specified quality
      const base64Image = canvas.toDataURL('image/jpeg', quality);
      
      resolve(base64Image);
    } catch (error) {
      reject(error);
    }
  });
};

/**
 * Uploads an image to the server
 * @param imageData Base64 string of the image
 * @param userId User ID
 * @param topic Current topic
 * @returns Promise that resolves when the upload is complete
 */
export const uploadScreenshot = async (
  imageData: string,
  userId: string,
  topic: string
): Promise<void> => {
  try {
    // Add the full URL with the server port
    const response = await fetch('http://localhost:8080/api/user/screenshot', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        image_data: imageData,
        topic: topic
      }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status}, ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Screenshot uploaded successfully:', data);
  } catch (error) {
    console.error('Error uploading screenshot:', error);
    throw error;
  }
};

// Add a test function 
export const testScreenshotUpload = async (userId: string): Promise<void> => {
  try {
    console.log("Testing screenshot upload with minimal data...");
    const response = await fetch('http://localhost:8080/api/user/screenshot', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        image_data: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACv/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVIP/2Q==",
        topic: "Test Topic"
      }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status}, ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Test screenshot upload successful:', data);
  } catch (error) {
    console.error('Test upload failed:', error);
  }
}; 