"""
RSQSim Rupture Geometry Handler
================================
Computes finite-rupture distances for seismic hazard calculations.

This module bridges RSQSim catalog data with GMPE distance requirements by:
1. Reading fault geometry (triangular patches)
2. Identifying participating patches per event
3. Computing site-to-rupture distances (Rrup, Rjb)

Usage:
    geometry = RSQSimGeometryReader('geometry.flt')
    catalog = RSQSimCatalogReader('catalog.eList', 'catalog.pList', 
                                   'catalog.dList', 'catalog.tList')
    
    for event_id in event_ids:
        rupture = catalog.get_event_rupture(event_id, geometry)
        rrup, rjb, nearest_patch = rupture.distance_to_site(site_lat, site_lon)
"""

import numpy as np
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging
from pyproj import Transformer
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class Triangle:
    """Single triangular fault patch"""
    patch_id: int
    vertices: np.ndarray  # Shape (3, 3): [v1, v2, v3] where each v = [x, y, z]
    rake: float
    slip_rate: float
    
    @property
    def centroid(self) -> np.ndarray:
        """Patch center point"""
        return np.mean(self.vertices, axis=0)
    
    @property
    def area(self) -> float:
        """Patch area in m^2"""
        v1, v2, v3 = self.vertices
        cross = np.cross(v2 - v1, v3 - v1)
        return 0.5 * np.linalg.norm(cross)
    
    def project_to_2d(self) -> np.ndarray:
        """Project triangle to surface (z=0) for Rjb calculation"""
        return self.vertices[:, :2]  # Just x, y coordinates


class RSQSimGeometryReader:
    """
    Reads RSQSim geometry.flt file containing triangular fault patches.
    
    File format (space/tab separated):
    x1 y1 z1 x2 y2 z2 x3 y3 z3 rake slip_rate [optional_cols...]
    
    Coordinates are in UTM (meters), z is depth (positive down).
    Patch IDs are 1-based (first line = patch 1).
    """
    
    def __init__(self, geometry_file: str):
        self.geometry_file = Path(geometry_file)
        self.patches: Dict[int, Triangle] = {}
        self._load_geometry()
    
    def _load_geometry(self):
        """Load all patches from geometry file"""
        logger.info(f"Loading geometry from {self.geometry_file}")
        
        with open(self.geometry_file, 'r') as f:
            patch_id = 1  # 1-based indexing
            
            for line in f:
                # Skip comments
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) < 11:
                    logger.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                
                # Parse vertex coordinates
                coords = [float(x) for x in parts[:9]]
                vertices = np.array(coords).reshape(3, 3)  # 3 vertices × 3 coords
                
                # Parse rake and slip_rate
                rake = float(parts[9])
                slip_rate = float(parts[10])
                
                self.patches[patch_id] = Triangle(
                    patch_id=patch_id,
                    vertices=vertices,
                    rake=rake,
                    slip_rate=slip_rate
                )
                patch_id += 1
        
        logger.info(f"Loaded {len(self.patches)} patches")
    
    def get_patch(self, patch_id: int) -> Triangle:
        """Get triangle by ID"""
        return self.patches[patch_id]
    
    def get_patches(self, patch_ids: List[int]) -> List[Triangle]:
        """Get multiple triangles"""
        return [self.patches[pid] for pid in patch_ids if pid in self.patches]


class RSQSimCatalogReader:
    """
    Reads RSQSim catalog list files (binary) to reconstruct event participation.
    
    Files are paired by index:
    - eList: event IDs (4-byte int)
    - pList: patch IDs (4-byte int)  
    - dList: slip on patch (8-byte double, meters)
    - tList: first-slip time (8-byte double, seconds)
    """
    
    def __init__(self, elist_file, plist_file, dlist_file, tlist_file):
        self.files = {
            'eList': Path(elist_file),
            'pList': Path(plist_file),
            'dList': Path(dlist_file),
            'tList': Path(tlist_file)
        }
        
        # Load everything ONCE at initialization
        self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog ONCE and index by event"""
        logger.info("Loading RSQSim catalog (one-time operation)...")
        
        # Read all files at once (much faster than chunked)
        event_ids = self._read_binary_ints(self.files['eList'])
        patch_ids = self._read_binary_ints(self.files['pList'])
        slips = self._read_binary_doubles(self.files['dList'])
        times = self._read_binary_doubles(self.files['tList'])
        
        logger.info(f"Read {len(event_ids):,} patch-event records")
        
        # Group by event using dict
        self.event_patches = {}
        self.event_slips = {}
        self.event_times = {}
        
        for i in range(len(event_ids)):
            eid = event_ids[i]
            pid = patch_ids[i]
            
            if eid not in self.event_patches:
                self.event_patches[eid] = []
                self.event_slips[eid] = {}
                self.event_times[eid] = {}
            
            self.event_patches[eid].append(pid)
            self.event_slips[eid][pid] = slips[i]
            self.event_times[eid][pid] = times[i]
        
        logger.info(f"Indexed {len(self.event_patches):,} unique events")
    
    @staticmethod
    def _read_binary_ints(filepath: Path) -> np.ndarray:
        """Read 4-byte little-endian integers"""
        with open(filepath, 'rb') as f:
            data = f.read()
        n = len(data) // 4
        return np.array(struct.unpack(f'<{n}i', data))
    
    @staticmethod
    def _read_binary_doubles(filepath: Path) -> np.ndarray:
        """Read 8-byte little-endian doubles"""
        with open(filepath, 'rb') as f:
            data = f.read()
        n = len(data) // 8
        return np.array(struct.unpack(f'<{n}d', data))
    
    def get_event_rupture(self, event_id: int, geometry):
        """Get pre-loaded rupture (instant lookup)"""
        if event_id not in self.event_patches:
            raise ValueError(f"Event {event_id} not found")
        
        patch_ids = self.event_patches[event_id]
        triangles = geometry.get_patches(patch_ids)
        
        return EventRupture(
            event_id=event_id,
            triangles=triangles,
            slips=self.event_slips[event_id],
            times=self.event_times[event_id]
        )


class EventRupture:
    """
    Represents a finite rupture surface for one earthquake event.
    Provides distance calculations to arbitrary site locations.
    """
    
    def __init__(self, event_id, triangles, slips, times, utm_zone=11):
        self.event_id = event_id
        self.triangles = triangles
        self.slips = slips  # patch_id -> slip (m)
        self.times = times  # patch_id -> first_slip_time (s)
        self.utm_zone = utm_zone
        self.latlon_to_utm = Transformer.from_crs(
            4326, f"326{utm_zone:02d}", always_xy=True  # For N hemisphere
        )
        
        # Pre-calculate bounding box for optimization
        if triangles:
            all_coords = np.vstack([tri.vertices for tri in triangles])
            self.bbox_min = all_coords.min(axis=0)  # [min_x, min_y, min_z]
            self.bbox_max = all_coords.max(axis=0)  # [max_x, max_y, max_z]
            self.centroid = (self.bbox_min + self.bbox_max) / 2
            self.bbox_diagonal = np.linalg.norm(self.bbox_max[:2] - self.bbox_min[:2])
        else:
            self.bbox_min = None
            self.bbox_max = None
            self.centroid = None
            self.bbox_diagonal = 0

    @lru_cache(maxsize=10000)
    def _cached_latlon_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        """Cache UTM conversions for repeated site queries"""
        return self.latlon_to_utm.transform(lon, lat)
    
    def distance_to_site_latlon(self, site_lat, site_lon):
        """Convenience wrapper that converts lat/lon to UTM with caching"""
        site_x, site_y = self._cached_latlon_to_utm(site_lat, site_lon)
        return self.distance_to_site(site_x, site_y, 0.0)
        
    @property
    def n_patches(self) -> int:
        """Number of participating patches"""
        return len(self.triangles)
    
    @property
    def total_area(self) -> float:
        """Total rupture area in m^2"""
        return sum(tri.area for tri in self.triangles)
    
    def distance_to_site(self, site_x: float, site_y: float, site_z: float = 0.0,
                        coordinate_system: str = 'utm') -> Tuple[float, float, int]:
        """
        Compute minimum distances from site to rupture surface.
        
        Args:
            site_x, site_y, site_z: Site coordinates (default z=0 for surface)
            coordinate_system: 'utm' (meters) or will add 'latlon' later
        
        Returns:
            rrup: Minimum 3D distance to rupture surface (km)
            rjb: Minimum horizontal distance (Joyner-Boore, km)
            nearest_patch_id: ID of closest patch
        """
        site = np.array([site_x, site_y, site_z])
        site_2d = np.array([site_x, site_y])
        
        # Bounding box optimization for distant sites
        if self.bbox_min is not None and self.bbox_diagonal > 0:
            bbox_center_2d = (self.bbox_min[:2] + self.bbox_max[:2]) / 2
            approx_dist = np.linalg.norm(site_2d - bbox_center_2d)
            
            # If site is far from rupture (>2× diagonal), use centroid approximation
            if approx_dist > 2.5 * self.bbox_diagonal:
                centroid_dist_3d = np.linalg.norm(site - self.centroid)
                centroid_dist_2d = np.linalg.norm(site_2d - self.centroid[:2])
                
                # Return approximation with first patch ID
                return (centroid_dist_3d / 1000.0, 
                       centroid_dist_2d / 1000.0, 
                       self.triangles[0].patch_id if self.triangles else 0)
        
        # Detailed calculation for nearby sites
        min_rrup = float('inf')
        min_rjb = float('inf')
        nearest_patch = None
        
        for tri in self.triangles:
            # Compute 3D distance (Rrup)
            dist_3d = self._point_to_triangle_3d(site, tri.vertices)
            if dist_3d < min_rrup:
                min_rrup = dist_3d
                nearest_patch = tri.patch_id
            
            # Compute 2D distance (Rjb)
            dist_2d = self._point_to_triangle_2d(site_2d, tri.project_to_2d())
            min_rjb = min(min_rjb, dist_2d)
        
        # Convert from meters to kilometers
        return min_rrup / 1000.0, min_rjb / 1000.0, nearest_patch
    
    @staticmethod
    def _point_to_triangle_3d(point: np.ndarray, 
                             triangle: np.ndarray) -> float:
        """
        Minimum 3D distance from point to triangle.
        
        Algorithm:
        1. Project point onto triangle plane
        2. If projection inside triangle: return perpendicular distance
        3. Else: return distance to nearest edge or vertex
        """
        v0, v1, v2 = triangle
        
        # Compute triangle normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        
        # Project point onto plane
        w = point - v0
        dist_to_plane = abs(np.dot(w, normal))
        projection = point - dist_to_plane * normal
        
        # Check if projection is inside triangle using barycentric coords
        v0_to_p = projection - v0
        dot00 = np.dot(edge2, edge2)
        dot01 = np.dot(edge2, edge1)
        dot02 = np.dot(edge2, v0_to_p)
        dot11 = np.dot(edge1, edge1)
        dot12 = np.dot(edge1, v0_to_p)
        
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Inside triangle if u >= 0, v >= 0, u + v <= 1
        if u >= 0 and v >= 0 and u + v <= 1:
            return dist_to_plane
        
        # Outside: find distance to nearest edge or vertex
        min_dist = min(
            np.linalg.norm(point - v0),
            np.linalg.norm(point - v1),
            np.linalg.norm(point - v2),
            EventRupture._point_to_segment(point, v0, v1),
            EventRupture._point_to_segment(point, v1, v2),
            EventRupture._point_to_segment(point, v2, v0)
        )
        return min_dist
    
    @staticmethod
    def _point_to_triangle_2d(point: np.ndarray, 
                             triangle: np.ndarray) -> float:
        """Minimum 2D distance (for Rjb)"""
        v0, v1, v2 = triangle
        
        # Project onto each edge and find minimum
        distances = [
            EventRupture._point_to_segment_2d(point, v0, v1),
            EventRupture._point_to_segment_2d(point, v1, v2),
            EventRupture._point_to_segment_2d(point, v2, v0),
            np.linalg.norm(point - v0),
            np.linalg.norm(point - v1),
            np.linalg.norm(point - v2)
        ]
        
        # Check if point inside triangle (then distance is 0)
        if EventRupture._point_in_triangle_2d(point, triangle):
            return 0.0
        
        return min(distances)
    
    @staticmethod
    def _point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """3D point-to-line-segment distance"""
        ab = b - a
        ap = point - a
        
        # Handle degenerate segment (zero length)
        ab_dot = np.dot(ab, ab)
        if ab_dot < 1e-10:  # Essentially a point, not a segment
            return np.linalg.norm(point - a)
        
        t = np.dot(ap, ab) / ab_dot
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return np.linalg.norm(point - closest)
    
    @staticmethod
    def _point_to_segment_2d(point: np.ndarray, a: np.ndarray, 
                            b: np.ndarray) -> float:
        """2D point-to-line-segment distance"""
        ab = b - a
        ap = point - a
        
        # Handle degenerate segment (zero length)
        ab_dot = np.dot(ab, ab)
        if ab_dot < 1e-10:  # Essentially a point, not a segment
            return np.linalg.norm(point - a)
        
        t = np.dot(ap, ab) / ab_dot
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return np.linalg.norm(point - closest)
    
    @staticmethod
    def _point_in_triangle_2d(point: np.ndarray, 
                             triangle: np.ndarray) -> bool:
        """Check if 2D point is inside 2D triangle"""
        v0, v1, v2 = triangle
        
        # Barycentric coordinate test
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
                   (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(point, v0, v1)
        d2 = sign(point, v1, v2)
        d3 = sign(point, v2, v0)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load geometry and compute distances
    # geometry = RSQSimGeometryReader('path/to/geometry.flt')
    # catalog = RSQSimCatalogReader('eList', 'pList', 'dList', 'tList')
    # 
    # event_id = 12345
    # rupture = catalog.get_event_rupture(event_id, geometry)
    # 
    # site_utm_x, site_utm_y = 400000, 3750000  # Example UTM coords
    # rrup, rjb, nearest_patch = rupture.distance_to_site(site_utm_x, site_utm_y)
    # 
    # print(f"Event {event_id}:")
    # print(f"  Participating patches: {rupture.n_patches}")
    # print(f"  Rrup: {rrup:.2f} km")
    # print(f"  Rjb: {rjb:.2f} km")
    # print(f"  Nearest patch: {nearest_patch}")
    
    passrint(f"  Rjb: {rjb:.2f} km")
    # print(f"  Nearest patch: {nearest_patch}")
    
    pass