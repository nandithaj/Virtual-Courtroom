import React, { useState } from 'react';
import { GoogleMap, LoadScript, Marker } from '@react-google-maps/api';

// Define the default center of the map and its zoom level
const containerStyle = {
    width: '100%',
    height: '100%',
};

const center = {
    lat: 9.9312, // Example latitude (Kochi)
    lng: 76.2673, // Example longitude
};

function MyMap({ onMapClick }) {
    const [clickedLocation, setClickedLocation] = useState(null);

    const handleMapClick = (event) => {
        const lat = event.latLng.lat();
        const lng = event.latLng.lng();
        const coordinates = { lat, lng };
        setClickedLocation(coordinates);
        onMapClick(coordinates); // Pass clicked coordinates to parent component
    };

    return (

        <LoadScript googleMapsApiKey="">
            <GoogleMap
                mapContainerStyle={containerStyle}
                center={center}
                zoom={10}
                onClick={handleMapClick} // Handle click event
            >
                {clickedLocation && <Marker position={clickedLocation} />}
            </GoogleMap>
        </LoadScript>
    );
}

export default MyMap;
