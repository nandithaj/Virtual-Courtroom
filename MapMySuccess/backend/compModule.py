import requests
import json
def get_restaurants_nearby(location, radius, api_key):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.types"
    }
    data = {
        "includedTypes": ["restaurant"],
        "locationRestriction": {
            "circle": {
                "center": {"latitude": float(location.split(",")[0]), "longitude": float(location.split(",")[1])},
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
        
        return restaurants
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    return []

def filter_restaurants_by_cuisine(restaurants, target_cuisines):
    """Filters restaurants that match any cuisine in the target list."""
    matching_restaurants = [
        r for r in restaurants if any(cuisine.lower() in [t.lower() for t in target_cuisines] for cuisine in r["cuisines"])
    ]
    return matching_restaurants, len(matching_restaurants)

def getCompCount(LOCATION, RADIUS, API_KEY , target_cuisines):
    #target cuisines is a lsit
    restaurants = get_restaurants_nearby(LOCATION, RADIUS, API_KEY)

    if restaurants:
        matching_restaurants, total_count = filter_restaurants_by_cuisine(restaurants, target_cuisines)

        total_count = len(matching_restaurants)
        cuisine_str = ", ".join(target_cuisines).capitalize()

        summary_dict = {
            "title": f"Restaurants serving {cuisine_str} ({total_count} found):",
            "restaurants": []
        }

        for r in matching_restaurants:
            summary_dict["restaurants"].append({
                "name": r["name"],
                "cuisines": r["cuisines"]
            })

        print(json.dumps(summary_dict))
        summary_dict = json.dumps(summary_dict)

        print(f"Total cout :{len(restaurants)} , same type  {total_count}")


        return total_count+1 , len(restaurants) , summary_dict
    else:
        return 0, 0 , {"title": "No restaurants found", "restaurants": []}



