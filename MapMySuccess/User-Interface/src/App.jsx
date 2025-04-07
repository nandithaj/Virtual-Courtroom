import { useState } from "react";
import MyMap from "./components/MyMap";
import "./App.css";

function App() {
    const [clickedCoordinates, setClickedCoordinates] = useState(null);
    const [cuisineType, setCuisineType] = useState("");
    const [expectedPrice, setExpectedPrice] = useState("");
    const [pText, setPText] = useState("");
    const [remarks, setRemarks] = useState("");
    const [compCounter, setCompCounter] = useState(0);
    const [summary, setSummary] = useState("");
    const handleMapClick = async (coordinates) => {
        setClickedCoordinates(coordinates);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    latitude: coordinates.lat,
                    longitude: coordinates.lng,
                    cuisine: cuisineType,
                    expected_price: expectedPrice,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            setPText(data["prediction"]);

            setRemarks("")
            setSummary("")

            setCompCounter(parseInt(data["same_type"], 10));
            console.log(data);

            if (parseInt(data["same_type"], 10) > 0) {

                setSummary(JSON.parse(data["summary"]));
            }
            if (data["remarks"] == "NOROAD") {
                setRemarks("No Roads Nearby Not Ideal Location !! ")
            }
        } catch (error) {
            console.error("Error sending request:", error);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gray-100">
            <nav className="bg-blue-600 text-white p-4 shadow-md">
                <h1 className="text-xl font-bold text-center">MapMySuccess</h1>
            </nav>

            <div className="flex flex-1 p-4 gap-4">
                <div className="w-1/3 bg-white p-6 rounded-xl shadow-lg">
                    <h4 className="text-lg font-semibold text-gray-700 mb-4">Success Score: <strong>{(parseFloat(pText) * 20).toFixed(2)}</strong></h4>
                    {remarks && <h4 className="text-lg font-semibold text-gray-700 mb-4">Remarks: <strong>{remarks}</strong></h4>}
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Cuisine Type</label>
                            <input
                                type="text"
                                value={cuisineType}
                                onChange={(e) => setCuisineType(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded-lg"
                                placeholder="Enter cuisine type"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Expected Price Range</label>
                            <input
                                type="text"
                                value={expectedPrice}
                                onChange={(e) => setExpectedPrice(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded-lg"
                                placeholder="Enter price range"
                            />
                        </div>
                    </div>

                    {clickedCoordinates && (
                        <div className="mt-6 p-4 bg-gray-50 rounded-lg shadow-md">
                            <h5 className="text-sm font-semibold">Clicked Coordinates:</h5>
                            <p className="text-sm">Latitude: {clickedCoordinates.lat}</p>
                            <p className="text-sm">Longitude: {clickedCoordinates.lng}</p>
                        </div>
                    )}
                    <div className="mt-6 bg-white p-4 rounded-lg shadow-md overflow-y-auto max-h-96">
                        {compCounter > 0 && summary && (
                            <div className="mt-6 bg-white p-4 rounded-lg shadow-md">
                                <h3 className="text-lg font-semibold text-blue-700 mb-2">{summary.title}</h3>
                                <ul className="space-y-2">
                                    {summary.restaurants.map((restaurant, index) => (
                                        <li
                                            key={index}
                                            className="border-l-4 border-blue-500 bg-blue-50 p-3 rounded-md shadow-sm"
                                        >
                                            <p className="font-medium text-gray-800">🔹 {restaurant.name}</p>
                                            <p className="text-sm text-gray-600">Cuisine(s): {restaurant.cuisines.join(", ")}</p>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>

                </div>

                <div className="flex-1 rounded-xl overflow-hidden shadow-lg">
                    <MyMap onMapClick={handleMapClick} />
                </div>
            </div>
        </div>
    );
}

export default App;
