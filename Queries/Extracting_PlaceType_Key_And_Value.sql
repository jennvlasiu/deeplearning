/* Extracting PlaceType Key and Value */

SELECT 
  StopLatitude,
  StopLongitude,
  PlaceLatitude,
  PlaceLongitude,
  replace(PlaceType, '=','') as PlaceTypeKey,
  SUBSTR(substr(Tags, STRPOS(Tags, PlaceType) + length(PlaceType)), 1, STRPOS(substr(Tags, STRPOS(Tags, PlaceType) + length(PlaceType)), '|')-1) as PlaceTypeValue,
  round(ST_Distance(ST_GeogPoint(StopLongitude, StopLatitude),ST_GeogPoint(PlaceLongitude, PlaceLatitude)),0) as Distance
FROM 
  `geotab-bi.Project_IndustryClassification.PlaceTypeAndTags`
