from .xr_analyzer.common import Units
from .xr_analyzer.xr_spatial_processor import XrSpatialProcessor
from .xr_analyzer.xr_zonal_stats import XrZonalStats
from .xr_analyzer.xr_geometry_processor import XrGeometryProcessor

# # Pro feature detection
# try:
#     from .pro import ingestor, datasets
#     HAS_PRO = True
# except ImportError:
#     HAS_PRO = False


# def requires_pro(func):
#     """Decorator to mark pro features."""
#     def wrapper(*args, **kwargs):
#         if not HAS_PRO:
#             raise ImportError(
#                 "This feature requires ctreeskit[pro]. "
#                 "Install with: pip install ctreeskit[pro]"
#             )
#         return func(*args, **kwargs)
#     return wrapper


__version__ = "0.1.0"
