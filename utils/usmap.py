"""Draw US state outlines from a small GeoJSON, with no mapping library.

Boundaries: `data/us_states_contiguous.geojson`, the contiguous states from the U.S. Census Bureau
cartographic boundaries (public domain), as redistributed in the PublicaMundi MappingAPI GeoJSON,
with coordinates rounded to three decimals.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "data" / "us_states_contiguous.geojson"


def albers(lon, lat, lon0=-96.0, lat0=37.5, phi1=29.5, phi2=45.5):
    """Albers equal-area conic projection (the usual choice for a US map). Inputs in degrees."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0, lat0, p1, p2 = map(np.radians, (lon0, lat0, phi1, phi2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    rho0 = np.sqrt(c - 2 * n * np.sin(lat0)) / n
    theta = n * (lon - lon0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


@lru_cache(maxsize=1)
def load_states(path=str(PATH)):
    """Return {state name: list of (n, 2) arrays of projected polygon outlines}."""
    if not Path(path).exists():
        return {}
    out = {}
    for f in json.loads(Path(path).read_text())["features"]:
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        rings = []
        for poly in polys:
            ring = np.array(poly[0], float)                  # outer ring only
            x, y = albers(ring[:, 0], ring[:, 1])
            rings.append(np.column_stack([x, y]))
        out[f["properties"]["name"]] = rings
    return out


def centroid(rings):
    """Area-weighted centroid of the largest ring of a state (shoelace formula)."""
    def area_centroid(r):
        x, y = r[:, 0], r[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        a = cross.sum() / 2
        return a, np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6 * a)
    best = max(rings, key=lambda r: abs(area_centroid(r)[0]))
    return area_centroid(best)[1]
