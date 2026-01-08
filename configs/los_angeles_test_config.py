from configs.los_angeles_optimized_config import load_optimized_config

def load_config(mode="sequential"):
    config = load_optimized_config(mode)
    
    # Override with small test grid (9 sites)
    config['site']['grid_lat_min'] = 34.0
    config['site']['grid_lat_max'] = 34.2
    config['site']['grid_lon_min'] = -118.5
    config['site']['grid_lon_max'] = -118.3
    config['site']['grid_lat_spacing'] = 0.1
    config['site']['grid_lon_spacing'] = 0.1
    
    return config
