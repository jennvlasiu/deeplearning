/* Consolidated Stops and Places (place type extracted) */

with places as
(
  SELECT ['aerialway=','aeroway=','amenity=','building=','craft=','emergency=','landuse=','leisure=','man_made=','military=','office=','power=','public_transport=','railway=','shop=','sport=','telecom=','tourism='] type
)

SELECT 
  StopLatitude,
  StopLongitude,
  PlaceType,
  PlaceLatitude,
  PlaceLongitude,
  x as Type
FROM 
  `geotab-bi.Project_IndustryClassification.PlaceTypes` P
  INNER JOIN UNNEST((select type from places)) x ON strpos(P.PlaceType,x) > 0

