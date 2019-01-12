select 
  T.*,
  S.PlaceLatitude,
  S.PlaceLongitude,
  S.PlaceTypeKey,
  S.PlaceTypeValue,
  S.Distance as DistanceBetweenStopAndPlace,
  H.IHS_CompanyVocation,
  H.IHS_CarrierType,
  H.IHS_GovVocation,
  H.DatabaseName
FROM
  `geotab-bigdata.Trips.Trips` T
INNER JOIN 
  `geotab-bi.Project_IndustryClassification.HardwareIndustryMapping` H 
  ON T.HardwareId=H.HardwareId
LEFT JOIN
  `geotab-bi.Project_IndustryClassification.StopLocationTypeDistance` S 
  ON T.StopLatitude=S.StopLatitude AND T.StopLongitude=S.StopLongitude

WHERE 
  T.StartTime >='2017-12-01' and T.StartTime < '2018-12-01'
