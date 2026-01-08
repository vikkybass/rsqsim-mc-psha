from scipy.spatial import cKDTree
import numpy as np
import logging

class SpatialCatalogIndex:
    """Spatially indexed earthquake catalog for efficient distance queries"""
    
    def __init__(self, events, max_distance_km=300.0):
        self.events = events
        self.max_distance_km = max_distance_km
        
        if events:
            # Build spatial index from coordinates
            coords = np.array([(event.lat, event.lon) for event in events])
            self.spatial_index = cKDTree(coords)
            logger.info(f"Built spatial index for {len(events)} events")
        else:
            self.spatial_index = None
            
    def find_nearby_events(self, site_lat, site_lon, max_distance_km=None):
        """Find events within distance using spatial index"""
        if self.spatial_index is None:
            return []
            
        if max_distance_km is None:
            max_distance_km = self.max_distance_km
            
        # Convert km to approximate degrees for initial filtering
        max_distance_deg = max_distance_km / 111.0
        
        # Query spatial index for candidates
        indices = self.spatial_index.query_ball_point([site_lat, site_lon], max_distance_deg)
        
        # Return candidate events
        return [self.events[i] for i in indices]