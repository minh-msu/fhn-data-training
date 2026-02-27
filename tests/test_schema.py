from pydantic import BaseModel

class Item(BaseModel):
    pickup_longitude: float
    pickup_latitude: float 
    dropoff_longitude: float 
    dropoff_latitude: float 
    passenger_count: float
    dayofweek: float
    day: float
    month: float 
    year: float
    hour: float
    minute: float 
    second: float
    distance: float
    bearing: float