# Raw data (not stored in git, about 25 GB)

Needed only for steps 1, 2, 3d and 3e. Steps 3–5 and the websites run from `data/processed/`.

| File in `data/raw/` | Source |
|---|---|
| `dunnett_solar_2020_WGS84.gpkg`, `dunnett_wind_2020_WGS84.gpkg` | Dunnett et al. 2020, https://doi.org/10.6084/m9.figshare.11310269 (inside `global_wind_solar_2020.zip`) |
| `GSA_GHI_9arcsec.tif` | Global Solar Atlas, World GHI GeoTIFF: https://globalsolaratlas.info/download/world |
| `gwa_ws100/<ISO3>_wind-speed_100m.tif` | Global Wind Atlas, one file per European country: `https://globalwindatlas.info/api/gis/country/<ISO3>/wind-speed/100` |
| `gwa_ws100_other/{USA,CHN,IND}_wind-speed_100m.tif` | Same Global Wind Atlas link, for the USA, China and India |
| `grid.gpkg` | Gridfinder: https://zenodo.org/records/3628142 |
| `gadm.gpkg` (optional) | Only if the website border caches are deleted; inside the Dunnett archive |

Elevation and slope are read online from the Copernicus DEM GLO-90 on AWS (`copernicus-dem-90m`);
results are cached in `data/processed/dem_points*.csv`.
