import requests
import time
import json
import math
from datetime import datetime, timezone, timedelta
import requests
import traceback
from compModule import getCompCount
GOOGLE_MAPS_API_KEY = 

# Load data from data.json
with open("data.json", 'r') as json_file:
    place_data = json.load(json_file)

def get_restaurants_nearby(lat,long, radius, GOOGLE_MAPS_API_KEY):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.types"
    }
    data = {
        "includedTypes": ["restaurant"],
        "locationRestriction": {
            "circle": {
                "center": {"latitude":lat, "longitude":long},
                "radius": radius
            }
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            raise Exception(f"Google API Error: {data['error']['message']}")
        
        restaurants = []
        for place in data.get("places", []):
            name = place.get("displayName", {}).get("text", "Unknown")
            types = place.get("types", [])
            
            cuisine = types[0]
            cuisines = []
            for tt in types:

                if "_restaurant" in tt:  # Check if it's a cuisine type
                    cuisine = tt.split("_restaurant")[0]  # Get text before '_restaurant'
                    cuisines.append(cuisine)
            if len(cuisines) == 0 :
                cuisines.append("restaurant")
                

            restaurants.append({"name": name, "cuisines": cuisines})
        
        for i in restaurants:
            print(i["cuisines"])
        print(restaurants)
        return restaurants

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
    
    return []

def filter_restaurants_by_cuisine(restaurants, target_cuisines):
    """Filters restaurants that match any cuisine in the target list."""
    matching_restaurants = [
        r for r in restaurants if any(cuisine.lower() in [t.lower() for t in target_cuisines] for cuisine in r["cuisines"])
    ]
    return matching_restaurants, len(matching_restaurants)



# 🔍 Example Usage



def get_nearby_establishments(lat, lng, radius=500):
    places = []
    url = f"https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "",
        "X-Goog-FieldMask": "places.displayName,places.types,places.id,places.priceLevel"
    }

    payload = {
        "maxResultCount": 20,  # Limit results to avoid excessive costs
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius
            }
        }
    }

    while True:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            places.extend(result.get("places", []))  # Add places to list

            # Check if there's a nextPageToken for more results
            next_page_token = result.get("nextPageToken")
            if next_page_token:
                payload["pageToken"] = next_page_token
                time.sleep(2)  # Required delay before requesting the next page
            else:
                break
        else:
            print(f"Error fetching places: {response.status_code} - {response.text}")
            break

    return places

def map_to_broader_category(place_types):
    # Normalize the keys in the JSON file by making them lowercase and replacing spaces with underscores
    normalized_json_keys = {key.lower().replace(" ", "_"): value for key, value in place_data.items()}

    # We'll check if any place types map directly to our normalized JSON categories
    for place_type in place_types:
        broader_category = normalized_json_keys.get(place_type, None)  # Use normalized key lookup
        if broader_category is not None:
            return place_type  # Return the first matched broader category
    return None


# Function to get numeric value for a place category from the normalized JSON data
def get_numeric_value_for_place(place_category):
    # Look up the place category in the normalized JSON data
    normalized_json_keys = {key.lower().replace(" ", "_"): value for key, value in place_data.items()}
    return normalized_json_keys.get(place_category, None)

def calculate_average_traffic(lat, lng, radius=500):
    # Define destinations for traffic checks (some random points within 500m)
    destinations = [
        {"latitude": lat + 0.005, "longitude": lng},
        {"latitude": lat - 0.005, "longitude": lng},
        {"latitude": lat, "longitude": lng + 0.005},
        {"latitude": lat, "longitude": lng - 0.005}
    ]

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "",
        "X-Goog-FieldMask": "routes.duration,routes.travelAdvisory,routes.legs"
    }

    # Set departure time to 1 minute in the future
    departure_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()

    total_time = 0
    count = 0

    for dest in destinations:
        payload = {
            "origin": {"location": {"latLng": {"latitude": lat, "longitude": lng}}},
            "destination": {"location": {"latLng": dest}},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "departureTime": departure_time  # Future timestamp
        }

        time.sleep(2)  # Avoid exceeding rate limits
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            if "routes" in result and result["routes"]:
                travel_time = result["routes"][0]["duration"]  # e.g., "300s"
                total_time += int(travel_time.replace("s", ""))  # Convert to seconds
                count += 1
        else:
            print(f"Error: {response.status_code} - {response.text}")

    return (total_time / count) / 60 if count > 0 else None  # Convert to minutes

# Function to calculate the distance between two latitude and longitude points (Haversine formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat/2) * math.sin(d_lat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2) * math.sin(d_lon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c  # Distance in kilometers
    return distance * 1000  # Convert to meters


def find_distance_to_nearest_main_road(lat, lng):
    url = f"https://roads.googleapis.com/v1/snapToRoads?path={lat},{lng}&key={""}&interpolate=false"
    response = requests.get(url)

    if response.status_code == 200:
        result = response.json()
        if 'snappedPoints' in result:
            nearest_road_location = result['snappedPoints'][0]['location']
            nearest_lat = nearest_road_location['latitude']
            nearest_lng = nearest_road_location['longitude']

            # Use Haversine formula to calculate distance
            distance = calculate_distance(lat, lng, nearest_lat, nearest_lng)
            print("distance",distance)
            return distance
        else:
            return None
    else:
        print("Error:", response.status_code, response.text)
        return None

def get_place_details(place_id, api_key):
    """
    Use the new Google Places API to get the details of a place by place_id.
    """
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "",
        "X-Goog-FieldMask": "priceLevel"
    }
    price_level = 0
    time.sleep(2)  # Avoid exceeding rate limits
    response = requests.get(url, headers=headers)
    #print(response.json())
    if response.status_code == 200:
        data = response.json()
        price_level_string = data.get('priceLevel', 'N/A')

        match price_level_string:
            case "PRICE_LEVEL_FREE":
                return 1  # Free
            case "PRICE_LEVEL_INEXPENSIVE":
                return 2  # Inexpensive
            case "PRICE_LEVEL_MODERATE":
                return 3  # Moderate
            case "PRICE_LEVEL_EXPENSIVE":
                return 4  # Expensive
            case "PRICE_LEVEL_VERY_EXPENSIVE":
                return 5  # Very Expensive
            case _:
                return 0  # Default for invalid or missing value

    else:
        print(f"Error fetching place details for place_id {place_id}: {response.status_code} - {response.text}")
        return 'N/A'
def get_place_details1(place_id, api_key,nearby_est):
    for place in nearby_est:
        if place_id == place.get('id'):

            price_level_string = place.get('priceLevel', 'N/A')

            match price_level_string:
                case "PRICE_LEVEL_FREE":
                    return 1  # Free
                case "PRICE_LEVEL_INEXPENSIVE":
                    return 2  # Inexpensive
                case "PRICE_LEVEL_MODERATE":
                    return 3  # Moderate
                case "PRICE_LEVEL_EXPENSIVE":
                    return 4  # Expensive
                case "PRICE_LEVEL_VERY_EXPENSIVE":
                    return 5  # Very Expensive
                case _:
                    return 0  # Default for invalid or missing value

        else:
            print(f"Error fetching place details for place_id {place_id}: ")
            return 'N/A'

def find_restaurant_details(lat,lng,cuisine):
    # Get nearby establishments within 1 km
    nearby_establishments = get_nearby_establishments(lat, lng)

    # Initialize a list to store unique place categories
    places = []

    # Initialize a list to store numeric values corresponding to place categories
    numbers = []

    # Initialize a list to store price levels of restaurants
    price_levels = []

    # Iterate through the nearby establishments and check their types against the JSON file
    for place in nearby_establishments:
        place_types = place.get('types', [])  # List of types for the place

        # Try to map the place types to a broader category (only once per place)
        broader_category = map_to_broader_category(place_types)

        if broader_category:
            # Add unique broader category to the places list
            if broader_category not in places:
                places.append(broader_category)

            # Get the numeric value for this broader category from the JSON file
            numeric_value = get_numeric_value_for_place(broader_category)
            #print("Number value ==== ",numeric_value)
            # Only add one numeric value per place
            if numeric_value is not None and broader_category not in numbers:
                numbers.append(int(numeric_value))  # Ensure numeric value is an integer
            
            ##

                ## same type restaurant code

            ##

    # Calculate the average population density based on the numbers list
    if numbers:
        Avg_population_density = sum(numbers) / len(numbers)
    else:
        Avg_population_density = 0

    # Get average traffic in the area
    avg_traffic = calculate_average_traffic(lat, lng)

    # Convert traffic time to a severity rating (1-5)
    traffic_severity = avg_traffic
    # Find distance to nearest main road
    distance_to_main_road = find_distance_to_nearest_main_road(lat, lng)

    # Calculate competitor presence for the given coordinates and restaurant type
    #competitor_presence = competitor_presence_for_location(lat, lng, restaurant_type)

    # if price_levels:
    #     average_price_level = sum(price_levels) / len(price_levels)
    # else:
    #     average_price_level = None  # No price levels available

    average_price_level = 1

    #print(Avg_population_density, traffic_severity, distance_to_main_road, average_price_level)
    # Return the calculated metrics

    cuisine_list = [item.strip() for item in cuisine.split(",")]

    same_type,total_type,summary = getCompCount(str(lat)+","+str(lng), 400, "" ,cuisine_list)

    return Avg_population_density, traffic_severity, distance_to_main_road, average_price_level,same_type,total_type,summary

#find_restaurant_details(10.021694448423816,76.31281276042853,"indian")

