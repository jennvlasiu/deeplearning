/* Consolidated Stops and Places (only tags) */

with places as
(
  SELECT ['aerialway=','aeroway=','amenity=','building=','craft=','emergency=','landuse=','leisure=','man_made=','military=','office=','power=','public_transport=','railway=','shop=','sport=','telecom=','tourism='] type
),
osm AS
(
  SELECT 
    tags,
    lat,
    lon
   FROM
    `geotab-bigdata.SpatialReferenceData.OsmNodes`
   WHERE
    tags <> '' AND EXISTS (SELECT * FROM UNNEST((select type from places)) AS x WHERE strpos(tags,x) > 0)
)

SELECT 
  S.StopLatitude,
  S.StopLongitude,
  N.Tags as PlaceType,
  N.Lat as PlaceLatitude,
  N.Lon AS PlaceLongitude
FROM 
  `geotab-bi.Project_IndustryClassification.TripStopLocations` S,
  osm N
where 
  ST_DWITHIN(ST_GeogPoint(S.StopLongitude, S.StopLatitude), ST_GeogPoint(N.Lon, N.Lat), 100)
 
