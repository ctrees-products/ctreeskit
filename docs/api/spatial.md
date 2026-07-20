# Spatial processing API

Reference for the geometry and raster tools in `xr_analyzer`. See the
{doc}`../guides/spatial` guide for worked examples.

```{eval-rst}
.. autoclass:: ctreeskit.GeometryData
   :members:

.. autofunction:: ctreeskit.process_geometry

.. autofunction:: ctreeskit.clip_ds_to_bbox

.. autofunction:: ctreeskit.clip_ds_to_geom

.. autofunction:: ctreeskit.reproject_match_ds

.. autofunction:: ctreeskit.create_area_ds_from_degrees_ds

.. autofunction:: ctreeskit.create_proportion_geom_mask
```

## Helpers

```{eval-rst}
.. autofunction:: ctreeskit.get_single_var_data_array

.. autofunction:: ctreeskit.get_flag_meanings

.. autofunction:: ctreeskit.agg_classified_mapped_da
```
