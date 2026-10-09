from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
import requests
import os
from dotenv import load_dotenv

load_dotenv()
GRAPHHOPPER_API_KEY = os.getenv("GRAPHHOPPER_API_KEY")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# GRAPHHOPPER API KEY
# ============================================================

# Put your CURRENT GraphHopper API key here.
# Do NOT upload this key to GitHub.


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "US_Accidents_March23_sample.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_compressed.pkl"
)

KMEANS_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "kmeans.pkl"
)

SCALER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    kmeans_model = joblib.load(
        KMEANS_MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_MODEL_PATH
    )

    print("Machine learning models loaded successfully.")

except Exception as e:

    print(
        "Error loading models:",
        e
    )

    rf_model = None
    kmeans_model = None
    scaler = None


# ============================================================
# LOAD ACCIDENT DATA
# ============================================================

try:

    accidents_df = pd.read_csv(
        DATA_PATH,
        nrows=100000
    )

    print(
        f"Accident dataset loaded: "
        f"{len(accidents_df)} rows"
    )

except Exception as e:

    print(
        "Error loading accident dataset:",
        e
    )

    accidents_df = pd.DataFrame()


# ============================================================
# PREPARE ACCIDENT DATA
# ============================================================

if not accidents_df.empty:

    if "Start_Time" in accidents_df.columns:

        accidents_df["Start_Time"] = pd.to_datetime(
            accidents_df["Start_Time"],
            errors="coerce"
        )

        accidents_df["Hour"] = (
            accidents_df["Start_Time"]
            .dt.hour
        )

        accidents_df["Day_of_Week"] = (
            accidents_df["Start_Time"]
            .dt.dayofweek
        )

        accidents_df["Month"] = (
            accidents_df["Start_Time"]
            .dt.month
        )

    if "Start_Lat" in accidents_df.columns:

        accidents_df["Start_Lat"] = pd.to_numeric(
            accidents_df["Start_Lat"],
            errors="coerce"
        )

    if "Start_Lng" in accidents_df.columns:

        accidents_df["Start_Lng"] = pd.to_numeric(
            accidents_df["Start_Lng"],
            errors="coerce"
        )

    accidents_df = accidents_df.dropna(
        subset=[
            "Start_Lat",
            "Start_Lng"
        ]
    )


# ============================================================
# US GEOCODING
# ============================================================

def geocode_us_location(location_name):

    if not location_name:

        return None, (
            "Location cannot be empty."
        )

    if not GRAPHHOPPER_API_KEY:

        return None, (
            "GraphHopper API key is missing. "
            "Add your current API key in app.py."
        )

    geocode_url = (
        "https://graphhopper.com/api/1/geocode"
    )

    params = {

        "q":
            location_name,

        "locale":
            "en",

        "limit":
            5,

        "countrycode":
            "US",

        "key":
            GRAPHHOPPER_API_KEY
    }

    try:

        response = requests.get(
            geocode_url,
            params=params,
            timeout=20
        )

    except requests.RequestException as e:

        return None, (
            f"Geocoding request failed: {e}"
        )

    if response.status_code != 200:

        return None, (
            "GraphHopper geocoding failed: "
            + response.text
        )

    try:

        data = response.json()

    except Exception:

        return None, (
            "Invalid response received "
            "from GraphHopper geocoding."
        )

    hits = data.get(
        "hits",
        []
    )

    if not hits:

        return None, (
            f'Could not find a US location for "{location_name}".'
        )

    # --------------------------------------------------------
    # Find a US result
    # --------------------------------------------------------

    for hit in hits:

        country_code = str(
            hit.get(
                "countrycode",
                ""
            )
        ).upper()

        country = str(
            hit.get(
                "country",
                ""
            )
        ).lower()

        if (
            country_code == "US"
            or
            country == "united states"
        ):

            point = hit.get(
                "point"
            )

            if (
                isinstance(point, dict)
                and
                "lat" in point
                and
                "lng" in point
            ):

                lat = float(
                    point["lat"]
                )

                lng = float(
                    point["lng"]
                )

                return (
                    lng,
                    lat
                ), None

    return None, (
        f'"{location_name}" was not recognized '
        "as a US location."
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return 6371.0 * c


# ============================================================
# FIND ACCIDENTS NEAR ROUTE
# ============================================================

def find_route_accidents(
    coordinates,
    radius_km=1.0
):

    if accidents_df.empty:

        return pd.DataFrame()

    if not coordinates:

        return pd.DataFrame()

    valid_coordinates = []

    for point in coordinates:

        if (
            not isinstance(point, (list, tuple))
            or len(point) < 2
        ):
            continue

        try:

            lng = float(point[0])
            lat = float(point[1])

            valid_coordinates.append(
                (lat, lng)
            )

        except Exception:

            continue

    if not valid_coordinates:

        return pd.DataFrame()

    # Sample route points

    max_points = 40

    if len(valid_coordinates) > max_points:

        indexes = np.linspace(
            0,
            len(valid_coordinates) - 1,
            max_points
        ).astype(int)

        sampled_coordinates = [
            valid_coordinates[i]
            for i in indexes
        ]

    else:

        sampled_coordinates = valid_coordinates

    accident_indexes = set()

    accident_lat = accidents_df[
        "Start_Lat"
    ].to_numpy()

    accident_lng = accidents_df[
        "Start_Lng"
    ].to_numpy()

    for lat, lng in sampled_coordinates:

        distances = haversine_distance(
            lat,
            lng,
            accident_lat,
            accident_lng
        )

        nearby_indexes = np.where(
            distances <= radius_km
        )[0]

        for index in nearby_indexes:

            accident_indexes.add(
                accidents_df.index[index]
            )

    if not accident_indexes:

        return pd.DataFrame()

    return accidents_df.loc[
        list(accident_indexes)
    ]


# ============================================================
# HISTORICAL ACCIDENT RISK
# ============================================================

def calculate_historical_risk(
    accident_density
):

    density = max(
        float(accident_density),
        0.0
    )

    if density <= 0:

        return 0.0

    center = 4500.0
    scale = 1800.0

    historical_risk = (

        100.0
        /
        (
            1.0
            +
            np.exp(
                -(density - center)
                /
                scale
            )
        )

    )

    historical_risk = min(
        max(
            historical_risk,
            0.0
        ),
        100.0
    )

    return round(
        historical_risk,
        2
    )


# ============================================================
# ROUTE RISK CALCULATION
# ============================================================

def calculate_route_risk(
    coordinates,
    distance_km,
    datetime_value
):

    nearby_accidents = (
        find_route_accidents(
            coordinates,
            radius_km=1.0
        )
    )

    # --------------------------------------------------------
    # ACCIDENT DENSITY
    # --------------------------------------------------------

    accident_count = len(
        nearby_accidents
    )

    distance_for_density = max(
        float(distance_km),
        0.1
    )

    accident_density = (
        accident_count
        /
        distance_for_density
    ) * 100.0

    accident_density = round(
        accident_density,
        1
    )

    # --------------------------------------------------------
    # HISTORICAL RISK
    # --------------------------------------------------------

    historical_risk = (
        calculate_historical_risk(
            accident_density
        )
    )

    # --------------------------------------------------------
    # SELECT DATE / TIME
    # --------------------------------------------------------

    hour = int(
        datetime_value.hour
    )

    day_of_week = int(
        datetime_value.dayofweek
    )

    month = int(
        datetime_value.month
    )

    is_weekend = int(
        day_of_week >= 5
    )

    # --------------------------------------------------------
    # DEFAULT WEATHER VALUES
    # --------------------------------------------------------

    temperature = 70.0
    humidity = 50.0
    pressure = 29.9
    visibility = 10.0
    wind_speed = 5.0

    # --------------------------------------------------------
    # HISTORICAL WEATHER
    # --------------------------------------------------------

    if not nearby_accidents.empty:

        if (
            "Temperature(F)"
            in nearby_accidents.columns
        ):

            value = pd.to_numeric(
                nearby_accidents[
                    "Temperature(F)"
                ],
                errors="coerce"
            ).dropna()

            if not value.empty:

                temperature = float(
                    value.mean()
                )

        if (
            "Humidity(%)"
            in nearby_accidents.columns
        ):

            value = pd.to_numeric(
                nearby_accidents[
                    "Humidity(%)"
                ],
                errors="coerce"
            ).dropna()

            if not value.empty:

                humidity = float(
                    value.mean()
                )

        if (
            "Pressure(in)"
            in nearby_accidents.columns
        ):

            value = pd.to_numeric(
                nearby_accidents[
                    "Pressure(in)"
                ],
                errors="coerce"
            ).dropna()

            if not value.empty:

                pressure = float(
                    value.mean()
                )

        if (
            "Visibility(mi)"
            in nearby_accidents.columns
        ):

            value = pd.to_numeric(
                nearby_accidents[
                    "Visibility(mi)"
                ],
                errors="coerce"
            ).dropna()

            if not value.empty:

                visibility = float(
                    value.mean()
                )

        if (
            "Wind_Speed(mph)"
            in nearby_accidents.columns
        ):

            value = pd.to_numeric(
                nearby_accidents[
                    "Wind_Speed(mph)"
                ],
                errors="coerce"
            ).dropna()

            if not value.empty:

                wind_speed = float(
                    value.mean()
                )

    # --------------------------------------------------------
    # ROUTE LOCATION
    # --------------------------------------------------------

    if coordinates:

        middle_point = coordinates[
            len(coordinates) // 2
        ]

        route_lng = float(
            middle_point[0]
        )

        route_lat = float(
            middle_point[1]
        )

    else:

        route_lat = 34.05
        route_lng = -117.65

    # --------------------------------------------------------
    # RANDOM FOREST FEATURES
    # --------------------------------------------------------

    feature_values = {

        "Start_Lat":
            route_lat,

        "Start_Lng":
            route_lng,

        "Distance(mi)":
            float(distance_km) * 0.621371,

        "Temperature(F)":
            temperature,

        "Humidity(%)":
            humidity,

        "Pressure(in)":
            pressure,

        "Visibility(mi)":
            visibility,

        "Wind_Speed(mph)":
            wind_speed,

        "Hour":
            hour,

        "Day_of_Week":
            day_of_week,

        "Month":
            month,

        "Is_Weekend":
            is_weekend,

        "Amenity":
            0,

        "Bump":
            0,

        "Crossing":
            0,

        "Give_Way":
            0,

        "Junction":
            0,

        "Railway":
            0,

        "Roundabout":
            0,

        "Station":
            0,

        "Stop":
            0,

        "Traffic_Calming":
            0,

        "Traffic_Signal":
            0
    }

    # --------------------------------------------------------
    # RANDOM FOREST PREDICTION
    # --------------------------------------------------------

    prediction = 2.0

    if rf_model is not None:

        try:

            expected_features = list(
                rf_model.feature_names_in_
            )

            feature_row = pd.DataFrame(

                [[
                    feature_values.get(
                        feature,
                        0
                    )

                    for feature
                    in expected_features

                ]],

                columns=expected_features

            )

            prediction = rf_model.predict(
                feature_row
            )[0]

            prediction = float(
                prediction
            )

            prediction = min(
                max(
                    prediction,
                    1.0
                ),
                4.0
            )

        except Exception as e:

            print(
                "Random Forest prediction error:",
                e
            )

            prediction = 2.0

    # --------------------------------------------------------
    # CONVERT ML SEVERITY TO 0-100 RISK
    # --------------------------------------------------------

    rf_risk = (
        (
            prediction - 1.0
        )
        /
        3.0
    ) * 100.0

    rf_risk = min(
        max(
            rf_risk,
            0.0
        ),
        100.0
    )

    # --------------------------------------------------------
    # FINAL RISK
    # --------------------------------------------------------

    final_risk = (

        historical_risk * 0.40

        +

        rf_risk * 0.60

    )

    final_risk = min(
        max(
            final_risk,
            0.0
        ),
        100.0
    )

    final_risk = round(
        final_risk,
        2
    )

    # --------------------------------------------------------
    # SAFETY SCORE
    # --------------------------------------------------------

    safety_score = (
        100.0
        -
        final_risk
    )

    safety_score = round(
        safety_score,
        2
    )

    return {

        "accident_density":
            accident_density,

        "historical_risk":
            round(
                historical_risk,
                2
            ),

        "rf_prediction":
            round(
                prediction,
                2
            ),

        "rf_risk":
            round(
                rf_risk,
                2
            ),

        "risk_score":
            final_risk,

        "safety_score":
            safety_score
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE ROUTE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        data = request.get_json()

        source = str(
            data.get(
                "source",
                "Los Angeles"
            )
        ).strip()

        destination = str(
            data.get(
                "destination",
                "Ontario"
            )
        ).strip()

        selected_date = data.get(
            "date"
        )

        selected_time = data.get(
            "time"
        )

        print(
            "Requested journey:",
            source,
            "->",
            destination
        )

        print(
            "Selected date:",
            selected_date
        )

        print(
            "Selected time:",
            selected_time
        )

        # ----------------------------------------------------
        # DATE + TIME
        # ----------------------------------------------------

        if (
            selected_date
            and
            selected_time
        ):

            datetime_value = pd.to_datetime(
                f"{selected_date} {selected_time}",
                errors="coerce"
            )

        else:

            datetime_value = pd.Timestamp.now()

        if pd.isna(
            datetime_value
        ):

            datetime_value = pd.Timestamp.now()

        # ----------------------------------------------------
        # US GEOCODING
        # ----------------------------------------------------

        print(
            "Geocoding FROM:",
            source
        )

        start, start_error = (
            geocode_us_location(
                source
            )
        )

        if start is None:

            return jsonify({

                "error":
                    f"FROM location error: "
                    f"{start_error}"

            }), 400

        print(
            "FROM coordinates:",
            start
        )

        print(
            "Geocoding TO:",
            destination
        )

        end, end_error = (
            geocode_us_location(
                destination
            )
        )

        if end is None:

            return jsonify({

                "error":
                    f"TO location error: "
                    f"{end_error}"

            }), 400

        print(
            "TO coordinates:",
            end
        )

        # ----------------------------------------------------
        # GRAPHHOPPER
        # ----------------------------------------------------

        if not GRAPHHOPPER_API_KEY:

            return jsonify({

                "error":
                    "GraphHopper API key is missing. "
                    "Add your current API key in app.py."

            }), 500

        route_url = (
            "https://graphhopper.com/api/1/route"
        )

        params = {

            "profile":
                "car",

            "algorithm":
                "alternative_route",

            "alternative_route.max_paths":
                3,

            "alternative_route.max_weight_factor":
                1.6,

            "alternative_route.max_share_factor":
                0.6,

            "locale":
                "en",

            "calc_points":
                "true",

            "points_encoded":
                "false",

            "instructions":
                "false",

            "point":
                [
                    f"{start[1]},{start[0]}",
                    f"{end[1]},{end[0]}"
                ],

            "key":
                GRAPHHOPPER_API_KEY
        }

        response = requests.get(
            route_url,
            params=params,
            timeout=30
        )

        if response.status_code != 200:

            return jsonify({

                "error":
                    "GraphHopper request failed.",

                "details":
                    response.text

            }), 500

        route_data = response.json()

        graphhopper_paths = (
            route_data.get(
                "paths",
                []
            )
        )

        if not graphhopper_paths:

            return jsonify({

                "error":
                    "No routes returned by GraphHopper."

            }), 500

        # ----------------------------------------------------
        # PROCESS ROUTES
        # ----------------------------------------------------

        route_results = []

        for index, path in enumerate(
            graphhopper_paths
        ):

            distance_km = (

                float(
                    path.get(
                        "distance",
                        0
                    )
                )
                /
                1000.0

            )

            travel_time_min = (

                float(
                    path.get(
                        "time",
                        0
                    )
                )
                /
                60000.0

            )

            coordinates = (

                path.get(
                    "points",
                    {}
                )
                .get(
                    "coordinates",
                    []
                )

            )

            risk_data = calculate_route_risk(

                coordinates,

                distance_km,

                datetime_value

            )

            route_results.append({

                "route_number":
                    index + 1,

                "distance_km":
                    round(
                        distance_km,
                        2
                    ),

                "travel_time_min":
                    round(
                        travel_time_min,
                        2
                    ),

                "coordinates":
                    coordinates,

                "accident_density":
                    risk_data[
                        "accident_density"
                    ],

                "historical_risk":
                    risk_data[
                        "historical_risk"
                    ],

                "rf_prediction":
                    risk_data[
                        "rf_prediction"
                    ],

                "rf_risk":
                    risk_data[
                        "rf_risk"
                    ],

                "risk_score":
                    risk_data[
                        "risk_score"
                    ],

                "safety_score":
                    risk_data[
                        "safety_score"
                    ]
            })

        # ----------------------------------------------------
        # FASTEST ROUTE
        # ----------------------------------------------------

        fastest_time = min(

            route[
                "travel_time_min"
            ]

            for route
            in route_results

        )

        # ----------------------------------------------------
        # FINAL SCORE
        #
        # 70% safety
        # 30% travel efficiency
        # ----------------------------------------------------

        for route in route_results:

            time_score = (

                fastest_time
                /
                route[
                    "travel_time_min"
                ]

            ) * 100.0

            time_score = min(
                max(
                    time_score,
                    0.0
                ),
                100.0
            )

            final_score = (

                route[
                    "safety_score"
                ] * 0.70

                +

                time_score * 0.30

            )

            route[
                "time_score"
            ] = round(
                time_score,
                2
            )

            route[
                "final_score"
            ] = round(
                final_score,
                2
            )

        # ----------------------------------------------------
        # RECOMMENDED ROUTE
        # ----------------------------------------------------

        recommended_route = max(

            route_results,

            key=lambda route:
                route[
                    "final_score"
                ]

        )

        # ----------------------------------------------------
        # FASTEST ROUTE
        # ----------------------------------------------------

        fastest_route = min(

            route_results,

            key=lambda route:
                route[
                    "travel_time_min"
                ]

        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "source":
                source,

            "destination":
                destination,

            "selected_date":
                str(
                    selected_date
                ),

            "selected_time":
                str(
                    selected_time
                ),

            "routes":
                route_results,

            "recommended":
                recommended_route,

            "recommended_route":
                recommended_route,

            "fastest_route":
                fastest_route
        })

    except Exception as e:

        print(
            "Analysis error:",
            e
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "        NAVIRA AI"
    )

    print(
        " Intelligent Journey. Safer Decisions."
    )

    print(
        "======================================"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "======================================\n"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )