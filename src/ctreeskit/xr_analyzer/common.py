from dataclasses import dataclass
from typing import Union, ClassVar


class MaskType:
    """Enumeration of mask types."""
    BINARY: str = "binary"
    WEIGHTED: str = "weighted"
    AREA: str = "area"


@dataclass
class AreaUnit:
    """Area unit configuration."""
    name: str
    conversion: float  # conversion factor from m²
    symbol: str


class Units:
    """Handle unit conversions for spatial calculations."""

    # Class-level constants
    M2: ClassVar[AreaUnit] = AreaUnit("square meters", 1.0, "m²")
    HA: ClassVar[AreaUnit] = AreaUnit("hectares", 0.0001, "ha")
    KM2: ClassVar[AreaUnit] = AreaUnit("square kilometers", 0.000001, "km²")

    _unit_map = {
        "m2": M2,
        "ha": HA,
        "km2": KM2
    }

    @classmethod
    def get_unit(cls, unit: Union[str, AreaUnit]) -> AreaUnit:
        """Get AreaUnit from string or AreaUnit input."""
        if isinstance(unit, AreaUnit):
            return unit
        if isinstance(unit, str) and unit.lower() in cls._unit_map:
            return cls._unit_map[unit.lower()]
        raise ValueError(
            f"Invalid unit: {unit}. Must be one of {list(cls._unit_map.keys())}")

    @classmethod
    def convert(cls, value: float, from_unit: AreaUnit, to_unit: AreaUnit) -> float:
        """Convert between area units."""
        return value * (from_unit.conversion / to_unit.conversion)
