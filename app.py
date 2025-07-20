from fastapi import FastAPI, HTTPException, Path, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pgeocode
import re
from functools import lru_cache
import logging
from typing import Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Setup
app = FastAPI(
    title="Global Postal Code to Location API",
    description="Supports postal codes from 80+ countries worldwide using FastAPI",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PostalCodeRequest(BaseModel):
    country_code: str = Field(..., description="2-letter ISO country code")
    postal_code: str = Field(..., description="Postal code in country format")

# Utilities
@lru_cache(maxsize=100)
def get_nominatim(country_code: str):
    try:
        return pgeocode.Nominatim(country_code.upper())
    except Exception as e:
        logger.error(f"Failed nominatim for {country_code}: {e}")
        return None

def safe_convert(value):
    if value is None or pd.isna(value): return None
    if isinstance(value, (np.integer, np.int64, np.int32)): return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)): return float(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if str(value).lower() in ['nan', '<na>', 'none']: return None
    return value

def validate_country_code(code: str):
    return bool(code and len(code) == 2 and code.isalpha())

def validate_postal_code(code: str, country: str):
    patterns = {
        'US': r'\d{5}(-\d{4})?','CA': r'^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$','GB': r'^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$','DE': r'^\d{5}$',
        'FR': r'^\d{5}$','JP': r'^\d{3}-\d{4}$','AU': r'^\d{4}$','IN': r'^\d{6}$','BR': r'^\d{5}-?\d{3}$','CN': r'^\d{6}$'
    }
    pattern = patterns.get(country.upper(), r'^[\w\d\s-]{3,10}$')
    return bool(re.match(pattern, code.strip()))

def format_coordinates(lat, lng):
    lat, lng = safe_convert(lat), safe_convert(lng)
    if lat and lng and -90 <= float(lat) <= 90 and -180 <= float(lng) <= 180:
        return {'latitude': round(float(lat), 6), 'longitude': round(float(lng), 6)}
    return None

def process_result(result, code, country):
    if result is None or result.empty or result.isna().all(): return None
    d = result.to_dict()
    data = {
        "postal_code": code,
        "country_code": country.upper(),
        "place_name": safe_convert(d.get('place_name')),
        "state_name": safe_convert(d.get('state_name')),
        "state_code": safe_convert(d.get('state_code')),
        "county_name": safe_convert(d.get('county_name')),
        "community_name": safe_convert(d.get('community_name')),
        "coordinates": format_coordinates(d.get('latitude'), d.get('longitude')),
        "accuracy": safe_convert(d.get('accuracy'))
    }
    return {k: v for k, v in data.items() if v is not None}

# Router
router = APIRouter(prefix="/v1")

@router.get("/location/{country_code}/{postal_code}")
async def get_location(country_code: str = Path(...), postal_code: str = Path(...)):
    if not validate_country_code(country_code):
        return {"success": False, "error": "Invalid country code", "data": None}
    if not validate_postal_code(postal_code, country_code):
        return {"success": False, "error": "Invalid postal code format", "data": None}
    nomi = get_nominatim(country_code)
    if not nomi:
        return {"success": False, "error": f"Unsupported country {country_code}", "data": None}
    result = nomi.query_postal_code(postal_code.strip())
    data = process_result(result, postal_code, country_code)
    if not data:
        return {"success": False, "error": "Postal code not found", "data": None}
    return {"success": True, "data": data}

@router.post("/location")
async def get_location_post(req: PostalCodeRequest):
    return await get_location(req.country_code, req.postal_code)

@router.get("/country")
async def get_supported_countries():
    supported_countries = {
        'AD': 'Andorra', 'AR': 'Argentina', 'AS': 'American Samoa', 'AT': 'Austria',
        'AU': 'Australia', 'BD': 'Bangladesh', 'BE': 'Belgium', 'BG': 'Bulgaria',
        'BR': 'Brazil', 'BY': 'Belarus', 'CA': 'Canada', 'CH': 'Switzerland',
        'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark', 'DO': 'Dominican Republic',
        'DZ': 'Algeria', 'ES': 'Spain', 'FI': 'Finland', 'FO': 'Faroe Islands',
        'FR': 'France', 'GB': 'United Kingdom', 'GF': 'French Guiana', 'GL': 'Greenland',
        'GP': 'Guadeloupe', 'GT': 'Guatemala', 'GU': 'Guam', 'HR': 'Croatia',
        'HU': 'Hungary', 'IE': 'Ireland', 'IN': 'India', 'IS': 'Iceland',
        'IT': 'Italy', 'JP': 'Japan', 'LI': 'Liechtenstein', 'LK': 'Sri Lanka',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MC': 'Monaco',
        'MD': 'Moldova', 'MH': 'Marshall Islands', 'MK': 'North Macedonia', 'MP': 'Northern Mariana Islands',
        'MQ': 'Martinique', 'MT': 'Malta', 'MX': 'Mexico', 'MY': 'Malaysia',
        'NC': 'New Caledonia', 'NL': 'Netherlands', 'NO': 'Norway', 'NZ': 'New Zealand',
        'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland', 'PM': 'Saint Pierre and Miquelon',
        'PR': 'Puerto Rico', 'PT': 'Portugal', 'RE': 'Reunion', 'RO': 'Romania',
        'RU': 'Russia', 'SE': 'Sweden', 'SI': 'Slovenia', 'SJ': 'Svalbard and Jan Mayen',
        'SK': 'Slovakia', 'SM': 'San Marino', 'TH': 'Thailand', 'TR': 'Turkey',
        'UA': 'Ukraine', 'US': 'United States', 'UY': 'Uruguay', 'VA': 'Vatican City',
        'VI': 'U.S. Virgin Islands', 'WF': 'Wallis and Futuna', 'YT': 'Mayotte',
        'ZA': 'South Africa'
    }

    return {
        "success": True,
        "supported_countries": supported_countries,
        "total_count": len(supported_countries)
    }

@router.get("/health")
async def health():
    return {"success": True, "data": {"status": "healthy", "version": "2.0"}}

# Mount router
app.include_router(router)

# For local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=2)
