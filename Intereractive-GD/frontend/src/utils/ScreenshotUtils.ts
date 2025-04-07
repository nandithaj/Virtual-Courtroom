
export interface Screenshot {
  timestamp: string;
  image_data: string;
}

/**
 * Fetches screenshots for a specific user from MongoDB
 * @param userId The ID of the user whose screenshots to fetch
 * @returns Promise that resolves to the user's screenshots data
 */
export const fetchUserScreenshots = async (
  userId: string
): Promise<Screenshot[]> => {
  try {
    const response = await fetch(
      `http://localhost:8080/api/user/${userId}/screenshots`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    console.log("API key ")

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status}, ${errorText}`);
    }

    const data = await response.json();
    if (!data.success) {
      throw new Error(data.error || "Failed to fetch screenshots");
    }

    return data.data.screenshots || [];
  } catch (error) {
    console.error("Error fetching screenshots:", error);
    throw error;
  }
};
